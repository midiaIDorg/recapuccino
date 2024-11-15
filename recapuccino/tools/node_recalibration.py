from functools import partial
from pathlib import Path

import click

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomllib
from pandas_ops.io import read_df, save_df
from recapuccino.importing import dynamically_import_foo
from recapuccino.xvalidation.splits import (
    create_group_assignments_at_random,
    find_optimal_models_using_xvalidation,
)


def save_figure(path, dpi=100, **kwargs) -> None:
    plt.savefig(path, dpi=dpi, **kwargs)
    plt.close()


def assert_config_fields_present(config, *fields):
    for field in fields:
        assert field in config, f"Config is missing field `{field}`."


refine_nodes_config_entries = [
    "monoisotopic_data",
    "Y",
    "precursor_X_def",
    "fragment_X_def",
    "model",
    "mz_recalibration_quantiles_cnt",
    "plot_settings",
    "x_validation_chunks_cnt",
]


@click.command(context_settings={"show_default": True})
@click.argument("filtered_precursors", type=Path)
@click.argument("filtered_matches", type=Path)
@click.argument("uncalibrated_precursor_stats", type=Path)
@click.argument("uncalibrated_fragment_stats", type=Path)
@click.argument("config", type=Path)
@click.argument("refined_precursor_stats", type=Path)
@click.argument("refined_fragment_stats", type=Path)
@click.argument("mz_recalibrated_distributions", type=Path)
@click.option("--quality_check_folder", type=Path, default=None)
@click.option("--verbose", is_flag=True)
def refine_nodes(
    filtered_precursors: Path,
    filtered_matches: Path,
    uncalibrated_precursor_stats: Path,
    uncalibrated_fragment_stats: Path,
    config: Path,
    refined_precursor_stats: Path,
    refined_fragment_stats: Path,
    mz_recalibrated_distributions: Path,
    quality_check_folder: Path | None = None,
    verbose: bool = False,
) -> None:
    """Refine nodes.

    Arguments:
        filtered_precursors (Path): Path to the already FDR (or score) filtered sage results.
        filtered_matches (Path): Path to the already FDR (or score) filtered and mapped back precursor-fragment edges.
        uncalibrated_precursor_stats (Path): Path to MS1 clusters' statistics.
        uncalibrated_fragment_stats (Path): Path to MS2 clusters' statistics.
        config (Path): Path to the config file (.toml).
        refined_precursor_stats (Path): Path to where to save the recalibated MS1 stats.
        refined_fragment_stats (Path): Path to where to save the recalibated MS2 stats.
        mz_recalibrated_distributions (Path): Path to where to save the recalibated m/z distributions (perhaps for the purpose of choosing new precursor and fragment search boxes?).
        quality_check_folder (Path): Path to where to save the quality plots for MS1 and MS2 m/z recalibration.
        verbose: bool = False
    """

    duck_conn = duckdb.connect()

    with open(config, "rb") as fh:
        config = tomllib.load(fh)

    assert_config_fields_present(config, *refine_nodes_config_entries)

    try:
        save_figure = partial(save_figure, dpi=config["plot_settings"]["dpi"])
    except KeyError:
        pass

    if verbose:
        print("Reading peptides.")

    monoisotopic_data = duck_conn.execute(
        config["monoisotopic_data"].format(results_sage=filtered_precursors)
    ).df()

    ms1 = read_df(uncalibrated_precursor_stats)
    sage_found_precursor_idxs = ms1.ClusterID.isin(
        monoisotopic_data.MS1_ClusterID.to_numpy()
    ).to_numpy()

    assert np.all(
        monoisotopic_data.MS1_ClusterID.isin(ms1.ClusterID)
    ), "Some clusters have wrong ids"

    ms1_sage_found = ms1.iloc[sage_found_precursor_idxs]
    ms1_sage_not_found = ms1.iloc[~sage_found_precursor_idxs]

    Y_precursor = monoisotopic_data[config["Y"]]
    X_precursor = duck_conn.execute(
        config["precursor_X_def"].format(ms1="ms1_sage_found")
    ).df()
    precursors_models_to_test = [
        (dynamically_import_foo(model["model_factory"]), model["hyperparameter"])
        for model in config["model"]
    ]

    if verbose:
        print("Performing ms1 model selection.")
    (
        optimal_precursor_models,
        predicted_precursor_ppm_diffs,
    ) = find_optimal_models_using_xvalidation(
        X=X_precursor,
        Y=Y_precursor,
        chunks_cnt=config["x_validation_chunks_cnt"],
        ModelFactory_hyperparameters_tuples=precursors_models_to_test,
    )

    X_precursor_new = duck_conn.execute(
        config["precursor_X_def"].format(ms1="ms1_sage_not_found")
    ).df()
    best_model_idx = np.argmin([x[1] for x in optimal_precursor_models])

    def update_stats(
        stats,
        optimal_precursor_model,
        sage_found_idxs,
        predicted_ppm_diffs,
        X_new,
    ):
        stats.rename(columns={"mz_wmean": "uncalibrated_mz_wmean"}, inplace=True)
        predicted_mz_diff_ppm = np.zeros(len(stats))
        predicted_mz_diff_ppm[sage_found_idxs] = predicted_ppm_diffs
        if len(X_new) > 0:
            predicted_mz_diff_ppm[~sage_found_idxs] = optimal_precursor_model.predict(
                X_new
            )
        if verbose:
            print("Updating mz_wmean.")
        stats["mz_wmean"] = calculate_mz_calibrated(
            predicted_mz_diff_ppm, stats.uncalibrated_mz_wmean
        )

    update_stats(
        stats=ms1,
        optimal_precursor_model=optimal_precursor_models[best_model_idx][0],
        sage_found_idxs=sage_found_precursor_idxs,
        predicted_ppm_diffs=predicted_precursor_ppm_diffs,
        X_new=X_precursor_new,
    )

    recalibrated_precursor_mz = ms1["mz_wmean"][sage_found_precursor_idxs].to_numpy()
    theory_precurosor_mz = monoisotopic_data.mz_calc.to_numpy()

    recalibrated_precursor_mz_ppm = (
        (theory_precurosor_mz - recalibrated_precursor_mz) / theory_precurosor_mz * 1e6
    )

    recalibrated_mz_distributions = pd.DataFrame(
        {"probability": np.linspace(0, 1, config["mz_recalibration_quantiles_cnt"])}
    )
    recalibrated_mz_distributions["ms1"] = np.quantile(
        recalibrated_precursor_mz_ppm, recalibrated_mz_distributions.probability
    )

    if quality_check_folder is not None:
        if verbose:
            print("Making plots for m/z precursor recalibration.")

        plt.hist(
            Y_precursor,
            bins=100,
            density=True,
            alpha=0.5,
            label="before recalibration",
        )
        plt.hist(
            recalibrated_precursor_mz_ppm,
            bins=100,
            density=True,
            alpha=0.5,
            label="after recalibration",
        )
        plt.legend()
        plt.xlabel("precursor m/z ppm")
        plt.ylabel("frequency")
        plt.title(" ".join(set(model["model_factory"] for model in config["model"])))
        save_figure(quality_check_folder / "precursor_recalibration.pdf")

        plt.scatter(
            theory_precurosor_mz,
            Y_precursor,
            s=config["plot_settings"]["point_size"],
            alpha=config["plot_settings"]["alpha"],
        )
        plt.xlabel("Theoretical M/Z of a precursor")
        plt.ylabel("M/Z theoretical minus experimental in ppm")
        save_figure(quality_check_folder / "precursor_mz_diff_as_function_of_mz.pdf")

    if verbose:
        print("Saving recalibated ms1.")

    save_df(ms1, refined_precursor_stats)

    if verbose:
        print("Reading fragments.")

    ms2 = read_df(uncalibrated_fragment_stats)

    if verbose:
        print("Parsing out fragments.")

    # we are preselcting only fragments mapped to monoisotopic precursors,
    # for they are without doubt also monoisotopic and it makes sense to direclty
    # compare one theoretical m/z with one experimental
    mapped_back_edges = read_df(filtered_matches).query(
        "MS1_ClusterID in @monoisotopic_data.MS1_ClusterID.to_numpy()"
    )

    sage_found_fragments = duck_conn.query(
        """
    SELECT 
        MS2_ClusterID,
        FIRST(fragment_mz_calculated) AS fragment_mz_calculated,
    FROM 'mapped_back_edges'
    GROUP BY MS2_ClusterID
    ORDER BY MS2_ClusterID
    """
    ).df()

    # sage_found_fragments = mapped_back_edges[[
    #     "MS2_ClusterID",
    #     "fragment_mz_calculated",
    # ]].drop_duplicates().sort_values("MS2_ClusterID", ignore_index=True)

    sage_found_ms2_ids = ms2.ClusterID.isin(sage_found_fragments.MS2_ClusterID)
    ms2_sage_found = ms2.loc[sage_found_ms2_ids]
    ms2_sage_not_found = ms2.loc[~sage_found_ms2_ids]

    # missing table def again fuck
    frag_mz_theo = sage_found_fragments.fragment_mz_calculated.to_numpy()
    frag_mz_exp = ms2_sage_found.mz_wmean.to_numpy()

    Y_fragment = pd.Series(
        (frag_mz_theo - frag_mz_exp) / frag_mz_theo * 1e6, name="mz_diff_ppm"
    )
    X_fragment = duck_conn.execute(
        config["fragment_X_def"].format(ms2="ms2_sage_found")
    ).df()

    fragments_models_to_test = [
        (dynamically_import_foo(model["model_factory"]), model["hyperparameter"])
        for model in config["model"]
    ]

    if verbose:
        print("Performing ms2 model selection.")
    (
        optimal_fragment_models,
        predicted_fragment_ppm_diffs,
    ) = find_optimal_models_using_xvalidation(
        X=X_fragment,
        Y=Y_fragment,
        chunks_cnt=config["x_validation_chunks_cnt"],
        ModelFactory_hyperparameters_tuples=fragments_models_to_test,
    )
    best_fragment_model_idx = np.argmin([x[1] for x in optimal_fragment_models])

    X_fragment_new = duck_conn.execute(
        config["fragment_X_def"].format(ms2="ms2_sage_not_found")
    ).df()

    update_stats(
        stats=ms2,
        optimal_precursor_model=optimal_fragment_models[best_fragment_model_idx][0],
        sage_found_idxs=sage_found_ms2_ids,
        predicted_ppm_diffs=predicted_fragment_ppm_diffs,
        X_new=X_fragment_new,
    )
    save_df(ms2, refined_fragment_stats)

    recalibrated_fragment_mz = ms2["mz_wmean"][sage_found_ms2_ids].to_numpy()
    theory_fragment_mz = frag_mz_theo

    recalibrated_fragment_mz_ppm = (
        (theory_fragment_mz - recalibrated_fragment_mz) / theory_fragment_mz * 1e6
    )
    recalibrated_mz_distributions["ms2"] = np.quantile(
        recalibrated_fragment_mz_ppm, recalibrated_mz_distributions.probability
    )

    if quality_check_folder is not None:
        if verbose:
            print("Plotting recalibrated vs original ms2 m/zs.")

        plt.hist(
            Y_fragment,
            bins=100,
            density=True,
            alpha=0.5,
            label="before recalibration",
        )
        plt.hist(
            recalibrated_fragment_mz_ppm,
            bins=100,
            density=True,
            alpha=0.5,
            label="after recalibration",
        )
        plt.legend()
        plt.xlabel("fragment m/z ppm")
        plt.ylabel("frequency")
        plt.title(" ".join(set(model["model_factory"] for model in config["model"])))
        save_figure(quality_check_folder / "fragment_recalibration.pdf")

        plt.scatter(
            theory_fragment_mz,
            Y_fragment,
            s=config["plot_settings"]["point_size"],
            alpha=config["plot_settings"]["alpha"],
        )
        plt.xlabel("Theoretical M/Z of a Fragment")
        plt.ylabel("M/Z theoretical minus experimental in ppm")
        save_figure(quality_check_folder / "fragment_mz_diff_as_function_of_mz.pdf")

        plt.plot(
            recalibrated_mz_distributions.ms1,
            recalibrated_mz_distributions.probability,
            label="precursor",
        )
        plt.plot(
            recalibrated_mz_distributions.ms2,
            recalibrated_mz_distributions.probability,
            label="fragment",
        )
        plt.title("Distribuants of recalibrated m/z distributions.")
        plt.legend()
        save_figure(quality_check_folder / "distribuant_of_recalibrated_mzs.pdf")

    save_df(
        recalibrated_mz_distributions,
        mz_recalibrated_distributions,
    )
    # OK, need to add in stats for quantiles: likely on the ppm level, cause to be used with SAGE.
