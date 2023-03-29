import sys

sys.path.append("../estimators")
sys.path.append("../basis")

import numpy as np

from basis.b_spline_basis import B_Spline_Basis
from estimators.base_estimator import Base_Estimator


class Smoothing_Spline_Estimator(Base_Estimator):
    def __init__(self, x, y, knots, order=3, lambda_=10e-3):

        self.x = x
        self.y = y
        self.name = "Smoothing_Spline_Estimator"

        self.knots = knots
        self.lambda_ = lambda_
        self.basis = B_Spline_Basis(x, knots, order=3)

    def fit(self, hypers: dict = None, beta=None):

        if beta is None:
            beta = self.basis.B(self.x)

        hypers["lambda_"]

        # regularization is used to improve numerical stability of the design matrix
        lhs = np.dot(beta.T, beta) + self.lambda_ * np.eye(beta.shape[1])
        L = np.linalg.cholesky(lhs)
        LT = L.T

        gamma = np.linalg.solve(L @ LT, beta.T.dot(self.y)).reshape(-1, 1)

        f_hat = beta.dot(gamma)

        return f_hat

    def estimate(self, t: np.ndarray):

        pass
