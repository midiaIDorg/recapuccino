from functools import partial
from math import inf

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer


def make_knots(xx: npt.NDArray, idx_thr: int, value_thr: float) -> npt.NDArray:
    """
    generate a list of x values at which nots should be placed for spline fitting
    :param xx: a list of x values like mz or retention time
    :param idx_thr: the minimum number of datapoints that should be placed into a segment fit
    :param value_thr: the minimum distance on the x axis between consecutive segments of the fit
    :return: a list of not placements
    """
    xx = np.sort(xx)
    knots = []
    prev_x = -inf
    prev_idx = -inf
    for idx, x in enumerate(xx):
        if (idx - prev_idx >= idx_thr) and (x - prev_x >= value_thr):
            knots.append(x)
            prev_idx = idx
            prev_x = x
    return np.array(knots).reshape(-1, 1)


class MannSplines:
    def __init__(
        self,
        retention_time_knots,
        mz_knots,
        fitter_factory=LinearRegression,
        rt_spline_kwargs={
            "extrapolation": "constant",
            "degree": 1,
            "include_bias": True,
        },
        mz_spline_kwargs={
            "extrapolation": "constant",
            "degree": 1,
            "include_bias": False,
        },
        fitter_factory_kwargs={"fit_intercept": False},
    ):
        self.rt_spline = SplineTransformer(
            knots=retention_time_knots.reshape(-1, 1),
            **rt_spline_kwargs,
        )
        self.mz_spline = SplineTransformer(
            knots=mz_knots.reshape(-1, 1),
            **mz_spline_kwargs,
        )
        self.fitter = fitter_factory(**fitter_factory_kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        assert "retention_time" in X.columns
        assert "mz" in X.columns
        X_rt = self.rt_spline.fit_transform(X.retention_time.to_numpy().reshape(-1, 1))
        X_mz = self.mz_spline.fit_transform(X.mz.to_numpy().reshape(-1, 1))
        X_tilde = np.hstack([X_rt, X_mz])
        self.fitter.fit(X_tilde, y)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        assert "retention_time" in X.columns
        assert "mz" in X.columns
        X_rt = self.rt_spline.fit_transform(X.retention_time.to_numpy().reshape(-1, 1))
        X_mz = self.mz_spline.fit_transform(X.mz.to_numpy().reshape(-1, 1))
        return pd.Series(self.fitter.predict(np.hstack([X_rt, X_mz])))
