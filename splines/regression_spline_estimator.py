import numpy as np
from b_spline_basis import B_Spline_Basis


class Regression_Spline_Estimator:

    """Regression Spline Estimator; Univariate Splines with Fixed Knot Set

    Developed in inspiration of Elements of Statistical Learning, Hastie, Tibshirani, Friedman, 2009, p. 185-186."""

    def __init__(self, x, y, knots, order=3, lambda_=10e-3):

        self.x = x
        self.y = y
        self.knots = knots
        self.lambda_ = lambda_
        self.basis = B_Spline_Basis(x, knots, order=3)

    def fit(self, beta=None):

        if beta is None:
            beta = self.basis.B(self.x)

        lhs = np.dot(beta.T, beta) + self.lambda_ * np.eye(beta.shape[1])
        L = np.linalg.cholesky(lhs)
        LT = L.T

        gamma = np.linalg.solve(L @ LT, beta.T.dot(self.y)).reshape(-1, 1)

        f_hat = beta.dot(gamma)

        return f_hat

    def predict(self, x: np.ndarray):

        beta = self.basis.B(x)

        lhs = np.dot(beta.T, beta) + self.lambda_ * np.eye(beta.shape[1])
        L = np.linalg.cholesky(lhs)
        LT = L.T

        gamma = np.linalg.solve(L @ LT, beta.T.dot(self.y)).reshape(-1, 1)

        return beta.dot(gamma)
