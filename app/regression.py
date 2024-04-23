import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import metrics

from supervised_learning.regression import Regression


def load_data(filename):
    with open(filename) as f:
        data = f.read()
    return data


def main():
    data = load_data('./data/real_estate.csv')
    r = Regression(n_iterations=100)
    n_features = 5
    r.initialize_weights_and_bias(n_features)
    print(f'w: {r.w}, b: {r.b}')


if __name__ == '__main__':
    main()
