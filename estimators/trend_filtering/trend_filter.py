import sys

sys.path.append("helpers/")
sys.path.append("../../basis")
from helpers.admm_tf import specialized_admm
from helpers.primal_dual_tf import primal_dual

from basis.continous_tf import Continous_TF
from matrix_algorithms.difference_matrix import Difference_Matrix


class Trend_Filter:
    def __init__(self, x, y, k, method="admm", lambda_=1.0):

        self.x = x
        self.y = y
        self.k = k

        # create difference matrix
        self.D = Difference_Matrix(len(y), k=k, t=x)

        # supported methods are admm or primal dual
        assert method in ["admm", "primal_dual"]

        self.method = method

        # initialize hyperparameters to be 1.0 wlog
        hypers = {"lambda_": lambda_}

        self.y_hat = self.fit(hypers)

    def fit(self, hypers: dict = None):

        assert hypers is not None and "lambda_" in hypers.keys()

        lambda_ = hypers["lambda_"]

        if self.method == "admm":

            y_hat = specialized_admm(self.y, self.D, lambda_)
        elif self.method == "primal_dual":

            y_hat = primal_dual(self.y, self.D, lambda_)
        else:
            raise ValueError("method not supported")

        return y_hat

    def estimate(self, x):
        """Estimates given a basis function"""
        return Continous_TF(self.x, self.y_hat, self.k).estimate(x)
