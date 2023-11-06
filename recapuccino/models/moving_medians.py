import typing

import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
from scipy.interpolate import interp1d
from scipy.signal import medfilt


def correct_delta_mzs_using_moving_medians_and_two_interpolations(
    observed_mz: npt.NDArray,
    delta_mz: npt.NDArray,
    f1_mz_grid: npt.NDArray = np.arange(0, 10_000),
    kernel_size_obs: int = 101,  # there were as many Dalmatines
    kernel_size_mz_grid: int = 101,  # there were as many Dalmatines
    f0_kwargs: dict = {"kind": "linear", "bounds_error": False, "fill_value": 0},
    f1_kwargs: dict = {"kind": "linear", "bounds_error": False, "fill_value": 0},
    return_all: bool = False,
) -> (
    typing.Callable
    | tuple[
        typing.Callable,
        typing.Callable,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
        npt.NDArray,
    ]
):
    """
    Args:
        observed_mz: strictly increasing sequence of observer mass to charge ratios.
        delta_mz: sequence of
    """
    assert np.all(
        np.diff(observed_mz) > 0
    ), "Passed in observed m/z must be strictly increasing."
    moving_medians = medfilt(delta_mz, kernel_size=kernel_size_obs)
    f0 = interp1d(observed_mz, moving_medians, **f0_kwargs)
    fitted_delta_values = f0(f1_mz_grid)
    moving_medians_mz_grid = medfilt(
        fitted_delta_values, kernel_size=kernel_size_mz_grid
    )
    f1 = interp1d(f1_mz_grid, moving_medians_mz_grid, **f1_kwargs)
    if return_all:
        return (
            f1,
            f0,
            observed_mz,
            delta_mz,
            f1_mz_grid,
            moving_medians,
            fitted_delta_values,
            moving_medians_mz_grid,
        )
    return f1


def plot_moving_medians_and_two_interpolactions(
    f1: typing.Callable,
    f0: typing.Callable,
    observed_mz: npt.NDArray,
    delta_mz: npt.NDArray,
    f1_mz_grid: npt.NDArray,
    moving_medians: npt.NDArray,
    fitted_delta_values: npt.NDArray,
    moving_medians_mz_grid: npt.NDArray,
    show=True,
) -> None:
    """For debugging results of 'correct_delta_mzs_using_moving_medians_and_two_interpolations' while 'return_all = True'."""
    import matplotlib.pyplot as plt

    mz_grid = np.linspace(min(observed_mz) - 10, max(observed_mz) + 10, 10000)
    plt.scatter(observed_mz, delta_mz, alpha=0.5, s=1)
    plt.plot(observed_mz, moving_medians, color="orange")
    plt.plot(observed_mz, f0(observed_mz), color="yellow")
    plt.scatter(f1_mz_grid, fitted_delta_values, color="green")
    plt.scatter(f1_mz_grid, moving_medians_mz_grid, color="green")
    plt.plot(mz_grid, f1(mz_grid), color="red")
    if show:
        plt.show()


class OldMovingMedians:
    def __init__(self):
        pass

    def fit(X: pd.DataFrame, y: pd.Series, *args, **kwargs):
        pass

    def predict(X):
        pass
