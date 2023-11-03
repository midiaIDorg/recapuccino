from functools import partial
from math import inf

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

from recapuccino.knots import make_knots


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
