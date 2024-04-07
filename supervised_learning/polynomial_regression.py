import numpy as np

from supervised_learning.regression import Regression


class PolynomialRegression(Regression):
    def __init__(self, n_iterations=20, learning_rate=0.00001, degree=3):
        """Polynomial regression model. Predicts the value of a target variable by learning a polynomial function."""
        super(PolynomialRegression, self).__init__(n_iterations=n_iterations, learning_rate=learning_rate)
        self.degree = degree

    def fit(self, X, y):
        """Fit the polynomial regression model to the training data."""
        X_poly = self._create_polynomial_features(X)
        super(PolynomialRegression, self).fit(X_poly, y)

    def predict(self, X):
        """Make predictions using the polynomial regression model."""
        X_poly = self._create_polynomial_features(X)
        return super(PolynomialRegression, self).predict(X_poly)

    def _create_polynomial_features(self, X):
        n_samples, n_features = X.shape
        X_poly = np.ones((n_samples, 1))
        for d in range(1, self.degree + 1):
            for i in range(n_features):
                X_poly = np.hstack((X_poly, (X[:, i] ** d).reshape(-1, 1)))
        return X_poly
