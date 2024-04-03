import math

import numpy as np


class Regression(object):
    def __init__(self, n_iterations=20, learning_rate=0.001):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.training_errors = []

    def initialize_weights_and_bias(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features,))
        self.b = 0

    def fit(self, X, y):
        n = len(y)
        for i in range(self.n_iterations):
            y_pred = X.dot(self.w) + self.b
            mse = np.mean(0.5 * (y - y_pred) ** 2)
            self.training_errors.append(mse)
            grad_w = -(y - y_pred).dot(X) / n
            grad_d = -(y - y_pred).sum() / n
            self.w -= grad_w * self.learning_rate
            self.b -= grad_d * self.learning_rate

    def predict(self, X):
        return X.dot(self.w) + self.b
