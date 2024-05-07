import numpy as np


class LossFunction:
    def loss(self, y, y_pred):
        raise NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()


class MeanSquaredError(LossFunction):
    def loss(self, y, y_pred):
        return 0.5 * np.mean(np.power(y - y_pred, 2))

    def gradient(self, y, y_pred):
        return y_pred - y


class CrossEntropy(LossFunction):

    def loss(self, y, y_pred):
        return - np.sum(y * np.log(y_pred))

    def gradient(self, y, y_pred):
        return y_pred - y