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
