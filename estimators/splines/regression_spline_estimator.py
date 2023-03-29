import sys

sys.path.append("../../estimators/")
sys.path.append("../../basis/")

import numpy as np
from b_spline_basis import B_Spline_Basis
from base_estimator import Base_Estimator


class Regression_Spline_Estimator(Base_Estimator):

    """Regression Spline Estimator; Univariate Splines with Fixed Knot Set

    Developed in inspiration of Elements of Statistical Learning, Hastie, Tibshirani, Friedman, 2009, p. 185-186."""

    def __init__(self, x, y, knots, degree, lambda_=10e-3):

        self.x = x
        self.y = y
        self.name = "Regression_Spline_Estimator"

        # regression spline specific params
        self.knots = knots
        self.lambda_ = lambda_
        self.degree = degree
        self.basis = B_Spline_Basis(x, knots, degree=degree)
        self.m = self.basis.m

        # fit the model
        self.gamma, self.y_hat = self.fit()
        self.num_params = self.gamma.shape[0]

    def fit(self, hypers: dict = None, beta=None):

        if beta is None:
            # always fetch the pth order basis functions
            beta = self.basis.B(self.x, m=self.degree + 1)

        # regularization is used to improve numerical stability of the design matrix
        lhs = np.dot(beta.T, beta) + self.lambda_ * np.eye(beta.shape[1])
        L = np.linalg.cholesky(lhs)
        LT = L.T

        gamma = np.linalg.solve(L @ LT, beta.T.dot(self.y)).reshape(-1, 1)

        f_hat = beta.dot(gamma)

        return gamma, f_hat

    def estimate(self, t: np.ndarray):

        beta = self.basis.B(t, m=self.degree + 1)

        gamma = self.gamma

        return beta.dot(gamma)
