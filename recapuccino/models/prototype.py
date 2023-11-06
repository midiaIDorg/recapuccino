import typing

import numpy.typing as npt
import pandas as pd


class Model(typing.Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, *args, **kwargs) -> None:
        ...

    def predict(self, X: pd.DataFrame) -> npt.NDArray:
        ...


class Preprocessing(typing.Protocol):
    def fit(self, data: pd.DataFrame) -> None:
        ...

    def transform(self, data: pd.DataFrame) -> npt.NDArray:
        ...
