from pathlib import Path

import click


@click.command(context_settings={"show_default": True})
@click.argument("found_precursors", type=Path)
@click.argument("found_matches", type=Path)
@click.argument("uncalibrated_precursor_stats", type=Path)
@click.argument("uncalibrated_fragment_stats", type=Path)
@click.argument("config", type=Path)
@click.argument("refined_precursor_stats", type=Path)
@click.argument("refined_fragment_stats", type=Path)
@click.argument("mz_recalibrated_distributions", type=Path)
@click.argument("quality_check_folder", type=Path)
@click.option("--verbose", is_flag=True)
def refine_nodes(
    found_precursors: Path,
    found_matches: Path,
    uncalibrated_precursor_stats: Path,
    uncalibrated_fragment_stats: Path,
    config: Path,
    refined_precursor_stats: Path,
    refined_fragment_stats: Path,
    mz_recalibrated_distributions: Path,
    quality_check_folder: Path,
    verbose: bool = False,
) -> None:
    pass
