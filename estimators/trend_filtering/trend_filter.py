from collections import defaultdict

import numpy as np

from basis.continous_tf import Continous_TF
from estimators.trend_filtering.helpers.admm_tf import specialized_admm
from estimators.trend_filtering.helpers.compute_lambda_max import compute_lambda_max
from estimators.trend_filtering.helpers.difference_matrix import Difference_Matrix
from estimators.trend_filtering.helpers.primal_dual_tf import primal_dual


class Trend_Filter:
    """
    Trend Filtering Estimator with support for inference with primal_dual or admm
    """

    def __init__(self, x, y, k, method="admm", prior=None):
        self.x = x
        self.y = y
        self.k = k
        self.prior = prior
        self.name = "Trend_Filter"

        # create difference matrix
        self.D = Difference_Matrix(len(y), k=k, t=x, prior=prior)

        # supported methods are admm or primal dual
        assert method in ["admm", "primal_dual"]

        self.method = method

        # initialize hyperparameters to be 0 for lambda
        self.hypers = defaultdict(float)
        self.hypers.update({"lambda_": 0})

        # init configs for model
        self.configs = {"k": k, "method": method}

        # set upper bound on hyperparameters
        self.lambda_max = compute_lambda_max(self.D, self.y)
        self.hyper_max = {"lambda_": self.lambda_max}

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
