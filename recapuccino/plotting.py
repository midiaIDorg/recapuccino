"""
Here we plot (against ourselves).
"""
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd


def plot_ppms(
    dataset: pd.DataFrame,
    show: bool = True,
    save_path: str | None = None,
    dpi: int = 50,
    **kwargs,
):
    """

    Arguments:
        **kwargs: key words to plt.savefig
    """
    ppm_error_cols = [c for c in dataset if c.split("_")[-1] == "ppm"]
    assert len(ppm_error_cols) > 0, "Nothing to plot."
    long_ppms_per_algo = dataset[ppm_error_cols].melt(
        var_name="Algorithm", value_name="mz_diff_in_ppm"
    )
    sns.kdeplot(
        data=long_ppms_per_algo,
        x="mz_diff_in_ppm",
        hue="Algorithm",
    )
    plt.grid()
    if save_path is not None:
        plt.savefig(fname=save_path, dpi=dpi)
        plt.close()
    if show:
        plt.show()
        plt.close()
