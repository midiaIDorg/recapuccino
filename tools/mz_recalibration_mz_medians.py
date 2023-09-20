#!/usr/bin/env python3

"""
recalibrating sage m/z results
"""
import argparse
import types
from pathlib import Path
from pprint import pprint

import duckdb
import numpy as np
import pandas as pd
from midia_search_engines.io import open_config

# from midia_search_engines.recalibration_ops import (
#     correct_delta_mzs_using_moving_medians_and_two_interpolations,
# )
from pandas_ops.io import save_df

# from parsers.sage import parse_fragments

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 10)


def in_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False


if in_ipython():

    def MockArgs():
        dataset = "G8602"
        calibration = "G8605"
        # dataset = "B5821"
        # calibration = "B5822"
        clusterer_1 = "08f35c4405b"
        clusterer_2 = "08f35c4405b"
        clusterer_config_1 = "default"
        clusterer_config_2 = "default"
        ms2_cluster_stats_script = "fast"
        ms2_stats_config = "default"
        matching_config = "wideRoughMatches"
        mgf_config = "0"
        sage_version = "95c2993"
        sage_config = "p12f15nd"
        fasta = "3"
        mz_recalibration_config = "default"
        return types.SimpleNamespace(
            MS1_stats_path=f"partial/{dataset}/{clusterer_1}/{clusterer_config_1}/ms1_cluster_stats.parquet",
            MS2_stats_path=f"partial/{dataset}/{calibration}/{clusterer_2}/{clusterer_config_2}/{ms2_cluster_stats_script}/{ms2_stats_config}/ms2_cluster_stats.parquet",
            results_sage=f"partial/{dataset}/{calibration}/{clusterer_1}/{clusterer_config_1}/{clusterer_2}/{clusterer_config_2}/{ms2_cluster_stats_script}/{ms2_stats_config}/{matching_config}/{mgf_config}/{sage_version}/{sage_config}/{fasta}/results.sage.tsv",
            config=f"configs/recalibration/mz/xgboost/{mz_recalibration_config}.toml",
            mz_recalibration_QC_plot_path="/tmp/mz_recalibration_QC.png",
            width=50,
            height=40,
            dpi=50,
            style="default",
            alpha=0.5,
            point_size=1,
            verbose=True,
        )

    args = MockArgs()
else:
    parser = argparse.ArgumentParser(
        "Return look-up tables with SAGE-based m/z recalibration for fragments and precursors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "MS1_stats_path",
        help="Path to MS1 (precursors) clusters' statistics.",
        type=Path,
    )
    parser.add_argument(
        "MS2_stats_path",
        help="Path to MS2 (fragments) clusters' statistics.",
        type=Path,
    )
    parser.add_argument(
        "results_sage",
        help="Path to the (likely) first generation SAGE results.",
        type=Path,
    )  # TODO: fucking generalize
    parser.add_argument(
        "config",
        help="Path to the config file.",
        type=Path,
    )
    parser.add_argument(
        "precursor_recalibrated",
        help="Path to where to save the recalibrated precursors' stats.",
        type=Path,
    )
    parser.add_argument(
        "fragment_recalibration",
        help="Path to where to save the recalibrated fragments' stats.",
        type=Path,
    )
    parser.add_argument(
        "--mz_recalibration_QC_plot_path",
        help="Path to where to save the quality plot for MS1 and MS2 m/z recalibration.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "-width",
        help="Plot width.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-height",
        help="Plot height.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-dpi",
        help="Plot dpi.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--style",
        help="Style of the plot.",
        default="default",
    )
    parser.add_argument(
        "-alpha",
        help="Transparency of the points.",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "-point_size",
        help="Size of points in the scatterplots.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--verbose",
        help="Show more output to stdout.",
        action="store_true",
    )
    args = parser.parse_args()


if args.verbose:
    pprint(args.__dict__)


if __name__ == "__main__":
    con = duckdb.connect()
    config = open_config(args.config)
    interpolation_mz_grid = np.arange(**config["mz_grid_np_arange_args"])

    if args.verbose:
        print("Reading peptides.")

    peptides = (
        con.execute(config["peptide_sql"].format(results_sage=args.results_sage))
        .df()
        .groupby("exp_mz")
        .calc_mz.first()
        .reset_index()
    )
    if args.verbose:
        print("Fitting peptide corrector.")
    precursor_corrector = correct_delta_mzs_using_moving_medians_and_two_interpolations(
        peptides.exp_mz,
        peptides.calc_mz - peptides.exp_mz,  # must remain `theory - experiment`!!!,
        f1_mz_grid=interpolation_mz_grid,
    )

    precursor_mz_calibration_lookup_table = pd.DataFrame(
        {
            "mz_original": interpolation_mz_grid,
            "mz_recalibrated": interpolation_mz_grid
            + precursor_corrector(interpolation_mz_grid),
        }
    )

    save_df(
        dataframe=precursor_mz_calibration_lookup_table,
        file_path=args.precursor_recalibration,
    )

    if args.verbose:
        print("Reading fragments.")
    fragments = con.execute(
        config["fragment_sql"].format(results_sage=args.results_sage)
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

            plt.savefig(
                args.mz_recalibration_QC_plot_path, dpi=args.dpi, transparent=False
            )
            plt.close()

    if args.verbose:
        print("m/z recalibration complete.")
