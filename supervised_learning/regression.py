import math

import numpy as np


class Regression(object):
    def __init__(self, n_iterations):
        self.n_iterations = n_iterations

    def initialize_weights_and_bias(self, n_features):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y):
        print(f'fit: X: {X}, y: {y}')
