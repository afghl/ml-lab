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
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self, x):
        p = self.__call__(x)
        grad = p * (1 - p)
        return grad
        # p = self.__call__(x)
        # # 创建一个空的张量来存储雅可比矩阵
        # grad = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        #
        # # 计算雅可比矩阵
        # for i in range(x.shape[0]):
        #     for j in range(x.shape[1]):
        #         for k in range(x.shape[1]):
        #             if j == k:
        #                 grad[i, j, k] = p[i, j] * (1 - p[i, j])
        #             else:
        #                 grad[i, j, k] = -p[i, j] * p[i, k]
        # return grad