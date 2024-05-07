import numpy as np


class LossFunction:
    def loss(self, y, y_pred):
        raise NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()


class MeanSquaredError(LossFunction):
    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return y_pred - y


class CrossEntropy(LossFunction):

    def loss(self, y, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y * np.log(p)

    def gradient(self, y, y_pred):
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return y_pred - y