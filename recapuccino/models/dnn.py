import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


class MzCorrectionDNNstandardized(tf.keras.Model):
    def __init__(
        self,
        compile_kwargs={
            "loss": "mean_absolute_error",
        },
        fit_kwargs={
            "epochs": 5,
            "batch_size": 128,
        },
        predict_kwargs={
            "batch_size": 128,
        },
        preprocessing: typing.Callable = StandardScaler(),
        preprocessing_kwargs: dict[str, typing.Any] | None = None,
        learning_rate: float = 1e-3,
        reg_factor: float = 0.001,
        **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.compile_kwargs = compile_kwargs
        self.kwarg = kwargs
        self.fit_kwargs = fit_kwargs
        self.predict_kwargs = predict_kwargs
        self.preprocessing = preprocessing
        self.reg_factor = reg_factor
        self.model_compiled = False

    def compile(self, input_shape: int):
        self.dense1 = tf.keras.layers.Dense(
            units=64,
            activation="relu",
            input_shape=(input_shape,),
            kernel_regularizer=tf.keras.regularizers.l1_l2(self.reg_factor),
        )
        self.dense2 = tf.keras.layers.Dense(
            units=64,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l1_l2(self.reg_factor),
        )
        self.dense3 = tf.keras.layers.Dense(1, activation="linear")
        super().compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            **self.compile_kwargs,
        )

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

    def fit(self, X: pd.DataFrame, y: pd.Series, *args, **kwargs):
        self.preprocessing.fit(X)
        if not self.model_compiled:
            self.compile(input_shape=X.shape[1])
        super().fit(
            x=self.preprocessing.transform(X),
            y=y.to_numpy(),
            **self.fit_kwargs,
        )

    def predict(self, X: pd.DataFrame) -> npt.NDArray:
        res = super().predict(
            x=self.preprocessing.transform(X),
            **self.predict_kwargs,
        )
        return res.flatten()


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.rand(1000, 6)
    coefficients = np.array([1.5, -2.3, 0.7, 4.2, -1.2, 3.1])
    y_true = X.dot(coefficients)
    noise = np.random.randn(1000)
    y = y_true + noise

    skaler = StandardScaler()
    X_scaled = skaler.fit_transform(X)

    dnn = MzCorrectionDNN()
    dnn.fit(X_scaled, y, epochs=5, batch_size=128)
    y_pred = dnn.predict(X_scaled)

    print("MAE: ", np.mean(np.abs(y_pred - y_true)))
