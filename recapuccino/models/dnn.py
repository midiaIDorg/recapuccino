import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

import tensorflow as tf


class MzCorrectionDNN(tf.keras.Model):
    def __init__(self, input_shape, learning_rate=1e-3, reg_factor=0.001, **kwargs):
        super(MzCorrectionDNN, self).__init__(**kwargs)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,),
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(reg_factor))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu',
                                            kernel_regularizer=tf.keras.regularizers.l1_l2(reg_factor))
        self.dense3 = tf.keras.layers.Dense(1, activation='linear')

        self.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


if __name__ == '__main__':
    np.random.seed(42)
    X = np.random.rand(1000, 6)
    coefficients = np.array([1.5, -2.3, 0.7, 4.2, -1.2, 3.1])
    y_true = X.dot(coefficients)
    noise = np.random.randn(1000)
    y = y_true + noise

    skaler = StandardScaler()
    X_scaled = skaler.fit_transform(X)

    dnn = MzCorrectionDNN(input_shape=6)
    dnn.fit(X_scaled, y, epochs=5, batch_size=128)
    y_pred = dnn.predict(X_scaled)

    print('MAE: ', np.mean(np.abs(y_pred - y_true)))