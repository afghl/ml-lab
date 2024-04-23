import numpy as np


class ReLU:
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)


class Identity:
    def __call__(self, x):
        return x

    def gradient(self, x):
        return np.ones_like(x)


class SoftMax:
    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        return p * (1 - p)
