#!/usr/bin/env python3
# %load_ext autoreload
# %autoreload 2

"""
recalibrating sage m/z results
"""
from pathlib import Path
from pprint import pprint

import patsy
import seaborn as sns
import toml

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_ops.io import read_df, save_df
from recapuccino.cli import common_cli, mock_args
from recapuccino.importing import dynamically_import_foo
from recapuccino.misc import in_ipython
from recapuccino.plotting import plot_ppms

DEBUG = False


def get_recalibration_errors(regressor, X, Y, mz_exps, mz_calcs):
    res = pd.DataFrame()
    res["mz_diff"] = Yhat = regressor.predict(np.array(X))
    res["error"] = Yhat - np.array(Y).flatten()
    res["mz_recalibrated"] = mz_exps + Yhat
    res["mz_diff_ppm"] = (mz_calcs - res.mz_recalibrated) / res.mz_recalibrated * 1e6
    return res


def get_results(data, formula, regressor):
    Y, X = patsy.dmatrices(formula, data)
    regressor.fit(np.array(X), np.array(Y))
    return get_results(regressor, X, Y, data.mz_exp, data.mz_calc)


if in_ipython():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 10)
    args = mock_args(
        dataset="G8602",
        calibration="G8605",
        # dataset="G8027",
        # calibration="G8045",
    )
else:
    parser = common_cli("Apply XGBoost based m/z-recalibration.")
    args = parser.parse_args()

# if __name__ == "__main__":


if args.verbose:
    pprint(args.__dict__)

if args.verbose:
    pprint("Reading peptides.")

config = toml.load(args.config)
con = duckdb.connect()
sage_precursors = con.execute(
    config["sage_results_for_recalibration"].format(
        results_sage=args.results_sage,
        max_FDR_level=config["max_FDR_level"],
    )
).df()
ms1_stats = read_df(args.MS1_stats_path)

precursors = pd.merge(
    sage_precursors,
    ms1_stats.iloc[sage_precursors.MS1_ClusterID - 1],
    left_on="MS1_ClusterID",
    right_on="ClusterID",
)

max_difference_between_IO_mz = np.abs(precursors.mz_exp - precursors.mz_wmean).max()
assert (
    max_difference_between_IO_mz < 0.001
), f"Observed a difference between SAGE reported precursor m/z and input mz_wmean beyond expected level of 0.001: was {max_difference_between_IO_mz}."


if args.plots_folder is not None:
    plt.scatter(
        precursors.mz_wmean, precursors.mz_exp - precursors.mz_wmean, s=1, alpha=0.5
    )
    plt.xlabel("Precursor mz_wmean")
    plt.ylabel("Precursor expamss/charge - mz_wmean")
    if DEBUG:
        plt.show()
    else:
        plt.savefig(args.plots_folder / "SAGE_IO_mz.pdf", dpi=args.dpi)
    plt.close()


if args.verbose:
    print("Fitting peptide corrector.")

# TODO generalize to what should be: a list of models with specified parameters.
Regressor = dynamically_import_foo(config["precursor_regressor"])
regressor = Regressor(**config["precursor_regressor_kwargs"])
y, X = patsy.dmatrices(config["precursor_regression_model"], precursors)
regressor.fit(X=X, y=y)

LHS_model = config["precursor_regression_model"].split("~")[1]
Xnew = patsy.dmatrix(LHS_model, ms1_stats)
ms1_stats["mz_wmean_original"] = ms1_stats.mz_wmean
ms1_stats["mz_wmean"] += regressor.predict(Xnew)

precursors["recalibrated_mz"] = precursors.mz_wmean + regressor.predict(X)
precursors["recalibrated_mz_diff_ppm"] = (
    (precursors.mz_calc - precursors.recalibrated_mz) / precursors.recalibrated_mz * 1e6
)


if args.plots_folder:
    plot_ppms(
        precursors,
        show=not DEBUG,
        save_path=args.plots_folder / "ppm_density.pdf",
        dpi=args.dpi,
    )

    plt.hexbin(
        ms1_stats.mz_wmean_original,
        (ms1_stats.mz_wmean - ms1_stats.mz_wmean_original)
        / ms1_stats.mz_wmean_original
        * 1e6,
        cmap="inferno",
    )
    plt.axhline(0, color="red")
    plt.xlabel("Original m/z")
    plt.ylabel("( Recalibrated m/z - Original m/z ) / Original m/z  [ppm] ")
    if DEBUG:
        plt.show()
    else:
        plt.savefig(
            fname=args.plots_folder / "Recalibrated_vs_Original_mz.pdf",
            dpi=args.dpi,
        )
    plt.close()
# now, read in all of stats and mod them?

if args.verbose:
    pprint("Saving Precursors.")

save_df(
    dataframe=precursors,
    file_path=args.output_MS1_stats_path,
)


if args.verbose:
    print("Reading fragments.")


fragments = con.execute(
    config["fragment_sql"].format(
        results_sage=args.results_sage, max_FDR_level=config["max_FDR_level"]
    )
).df()
fragments = (
    parse_fragments(fragments, fragment_types={c[0] for c in fragments.columns})
    .groupby("exp_mz")
    .first()
    .reset_index()
)

if args.verbose:
    print("Fitting fragment corrector.")
fragment_corrector = correct_delta_mzs_using_moving_medians_and_two_interpolations(
    fragments.exp_mz,
    fragments.calc_mz - fragments.exp_mz,  # must remain `theory - experiment`!!!
    f1_mz_grid=interpolation_mz_grid,
)

fragment_mz_calibration_lookup_table = pd.DataFrame(
    {
        "mz_original": interpolation_mz_grid,
        "mz_recalibrated": interpolation_mz_grid
        + fragment_corrector(interpolation_mz_grid),
    }
)

save_df(
    dataframe=fragment_mz_calibration_lookup_table,
    file_path=args.fragment_recalibration,
)

# QC plot
if args.mz_recalibration_QC_plot_path is not None:
    if args.verbose:
        print("Making QC plots.")
    import matplotlib.pyplot as plt

    with plt.style.context(args.style):
        fig, axs = plt.subplots(3, 2)
        fig.set_size_inches(args.width, args.height)
        axs[0, 0].scatter(
            peptides.exp_mz,
            peptides.calc_mz - peptides.exp_mz,
            alpha=args.alpha,
            s=args.point_size,
        )
        mz_grid = np.linspace(
            max(peptides.exp_mz.iloc[0] - 10, 1),
            peptides.exp_mz.iloc[-1] + 10,
            10_000,
        )
        axs[0, 0].plot(
            mz_grid,
            precursor_corrector(mz_grid),
            c="orange",
            label="precursor",
        )
        axs[0, 0].set_xlabel("Real precursor m/z")
        axs[0, 0].set_ylabel("Calculated m/z - Real m/z")

        axs[0, 1].scatter(
            peptides.exp_mz,
            (peptides.calc_mz - peptides.exp_mz) / peptides.exp_mz * 1e6,
            alpha=args.alpha,
            s=args.point_size,
            c="red",
        )
        axs[0, 1].plot(
            mz_grid,
            precursor_corrector(mz_grid) / mz_grid * 1e6,
            c="blue",
            label="fragment",
        )
        axs[0, 1].set_xlabel("Real precursor m/z")
        axs[0, 1].set_ylabel("( Calculated m/z - Real m/z ) / Real m/z [ppm]")

        axs[1, 0].scatter(
            fragments.exp_mz,
            fragments.calc_mz - fragments.exp_mz,
            alpha=args.alpha,
            s=args.point_size,
            c="red",
        )
        mz_grid = np.linspace(
            max(fragments.exp_mz.iloc[0] - 10, 1),
            fragments.exp_mz.iloc[-1] + 10,
            10_000,
        )
        axs[1, 0].plot(
            mz_grid,
            fragment_corrector(mz_grid),
            c="blue",
            label="fragment",
        )
        axs[1, 0].set_xlabel("Real fragment m/z")
        axs[1, 0].set_ylabel("Calculated m/z - Real m/z")

        axs[1, 1].scatter(
            fragments.exp_mz,
            (fragments.calc_mz - fragments.exp_mz) / fragments.exp_mz * 1e6,
            alpha=args.alpha,
            s=args.point_size,
            c="red",
        )
        axs[1, 1].plot(
            mz_grid,
            fragment_corrector(mz_grid) / mz_grid * 1e6,
            c="blue",
            label="fragment",
        )
        axs[1, 1].set_xlabel("Real fragment m/z")
        axs[1, 1].set_ylabel("( Calculated m/z - Real m/z ) / Real m/z [ppm]")

        mz_grid = np.linspace(
            max(
                min(
                    peptides.exp_mz.iloc[0],
                    fragments.exp_mz.iloc[0],
                )
                - 10,
                1,
            ),
            max(
                peptides.exp_mz.iloc[-1],
                fragments.exp_mz.iloc[-1],
            )
            + 10,
            10_000,
        )

        axs[2, 0].plot(
            mz_grid,
            precursor_corrector(mz_grid),
            c="orange",
            label="precursor",
        )
        axs[2, 0].plot(
            mz_grid,
            fragment_corrector(mz_grid),
            c="blue",
            label="fragment",
        )
        axs[2, 0].legend()
        axs[2, 0].set_xlabel("Real m/z")
        axs[2, 0].set_ylabel("Calculated m/z - Real m/z")

        axs[2, 1].plot(
            mz_grid,
            precursor_corrector(mz_grid) / mz_grid * 1e6,
            c="orange",
            label="precursor",
        )
        axs[2, 1].plot(
            mz_grid,
            fragment_corrector(mz_grid) / mz_grid * 1e6,
            c="blue",
            label="fragment",
        )
        axs[2, 1].legend()
        axs[2, 1].set_xlabel("Real m/z")
        axs[2, 1].set_ylabel("( Calculated m/z - Real m/z ) / Real m/z [ppm]")

        plt.savefig(args.mz_recalibration_QC_plot_path, dpi=args.dpi, transparent=False)
        plt.close()

if args.verbose:
    print("m/z recalibration complete.")
