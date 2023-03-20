import sys

import numpy as np

sys.path.append("..")

from splines.b_spline_basis import B_Spline_Basis


class Regression_Spline_Estimator:

    """Regression Spline Estimator; Univariate Splines with Fixed Knot Set

    Developed in inspiration of Elements of Statistical Learning, Hastie, Tibshirani, Friedman, 2009, p. 185-186."""

    def __init__(self, x, y, knots, order, lambda_=10e-3):

        self.x = x
        self.y = y
        self.knots = knots
        self.lambda_ = lambda_
        self.order = order
        self.basis = B_Spline_Basis(x, knots, order=order)

        self.gamma, self.f_hat = self.fit()

        self.num_params = self.gamma.shape[0]

    def fit(self, beta=None):

        if beta is None:
            # always fetch the pth order basis functions
            beta = self.basis.B(self.x, m=self.order + 1)

        # regularization is used to improve numerical stability of the design matrix
        lhs = np.dot(beta.T, beta) + self.lambda_ * np.eye(beta.shape[1])
        L = np.linalg.cholesky(lhs)
        LT = L.T

        gamma = np.linalg.solve(L @ LT, beta.T.dot(self.y)).reshape(-1, 1)

        f_hat = beta.dot(gamma)

        return gamma, f_hat

    def predict(self, x: np.ndarray):

        beta = self.basis.B(x, m=self.order + 1)

        gamma = self.gamma

        return beta.dot(gamma)
