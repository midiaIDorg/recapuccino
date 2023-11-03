import typing


class Model(typing.Protocol):
    def fit(X, y, *args, **kwargs):
        ...

    def predict(X):
        ...
