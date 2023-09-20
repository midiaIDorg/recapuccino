import argparse
import types
from pathlib import Path


def mock_args(
    dataset="G8602",
    calibration="G8605",
    # dataset = "B5821"
    # calibration = "B5822"
    clusterer_1="08f35c4405b",
    clusterer_2="08f35c4405b",
    clusterer_config_1="default",
    clusterer_config_2="default",
    ms2_cluster_stats_script="fast",
    ms2_stats_config="default",
    matching_config="wideRoughMatches",
    mgf_config="0",
    sage_version="95c2993",
    sage_config="p12f15nd",
    fasta="3",
    mz_recalibration_config="default",
):
    res = types.SimpleNamespace(
        MS1_stats_path=Path(
            f"partial/{dataset}/{clusterer_1}/{clusterer_config_1}/ms1_cluster_stats.parquet"
        ),
        MS2_stats_path=Path(
            f"partial/{dataset}/{calibration}/{clusterer_2}/{clusterer_config_2}/{ms2_cluster_stats_script}/{ms2_stats_config}/ms2_cluster_stats.parquet"
        ),
        results_sage=Path(
            f"partial/{dataset}/{calibration}/{clusterer_1}/{clusterer_config_1}/{clusterer_2}/{clusterer_config_2}/{ms2_cluster_stats_script}/{ms2_stats_config}/{matching_config}/{mgf_config}/{sage_version}/{sage_config}/{fasta}/results.sage.tsv"
        ),
        config=Path(f"configs/recalibration/mz/xgboost/{mz_recalibration_config}.toml"),
        output_MS1_stats_path=Path("/tmp/ms1_clusters_stats.parquet"),
        output_MS2_stats_path=Path("/tmp/ms2_clusters_stats.parquet"),
        plots_folder=Path("/tmp/recalibratoin_plots"),
        width=50,
        height=40,
        dpi=50,
        style="default",
        alpha=0.5,
        point_size=1,
        verbose=True,
    )
    res.plots_folder.mkdir(exist_ok=True, parents=True)
    return res


def common_cli(main_message):
    parser = argparse.ArgumentParser(
        main_message,
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
        "output_MS1_stats_path",
        help="Path to where to save the recalibrated precursors' stats.",
        type=Path,
    )
    parser.add_argument(
        "output_MS2_stats_path",
        help="Path to where to save the recalibrated fragments' stats.",
        type=Path,
    )
    parser.add_argument(
        "--plots_folder",
        help="Path to where to save the quality plot for MS1 and MS2 m/z recalibration.",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--width",
        help="Plot width.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--height",
        help="Plot height.",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--dpi",
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
        "--alpha",
        help="Transparency of the points.",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--point_size",
        help="Size of points in the scatterplots.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--verbose",
        help="Show more output to stdout.",
        action="store_true",
    )
    return parser
