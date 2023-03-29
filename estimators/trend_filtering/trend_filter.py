import sys

sys.path.append("helpers/")
sys.path.append("../../basis")

from collections import defaultdict

import numpy as np
from helpers.admm_tf import specialized_admm
from helpers.compute_lambda_max import compute_lambda_max
from helpers.difference_matrix import Difference_Matrix
from helpers.primal_dual_tf import primal_dual

from basis.continous_tf import Continous_TF


class Trend_Filter:
    def __init__(self, x, y, k, method="admm"):

        self.x = x
        self.y = y
        self.k = k
        self.name = "Trend_Filter"

        # create difference matrix
        self.D = Difference_Matrix(len(y), k=k, t=x)

        # supported methods are admm or primal dual
        assert method in ["admm", "primal_dual"]

        self.method = method

        # initialize hyperparameters to be 0 for lambda
        self.hypers = defaultdict(float)
        self.hypers.update({"lambda_": 0})

        self.lambda_max = compute_lambda_max(self.D, self.y)

        self.hyper_max = {"lambda_": self.lambda_max}

        # fit base model
        self.y_hat = self.fit()

    def fit(self, warm_start=False):
        """Fit estimator to data given hyperparameters"""

        assert self.hypers is not None and "lambda_" in self.hypers.keys()

        lambda_ = self.hypers["lambda_"]

        if self.method == "admm":

            y_hat = specialized_admm(self.y, self.D, lambda_, initial_guess=self.y_hat if warm_start else None)
        elif self.method == "primal_dual":

            y_hat = primal_dual(self.y, self.D, lambda_, initial_guess=self.y_hat if warm_start else None)
        else:
            raise ValueError("method not supported")

        return y_hat

    def estimate(self, t: np.ndarray):
        """Estimates given a basis function"""
        return Continous_TF(self.y_hat, self.D, self.k).evaluate_tf(t)

    def update_params(self, hypers: dict):
        """Update parameters of estimator"""
        self.hypers.update(hypers)

        self.y_hat = self.fit()

        return
