from abc import ABC

import numpy as np


class Base_Estimator(ABC):
    """
    Base class for all estimators; which have a fit and estimate method
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.y_hat = None

    def fit(self, hypers: dict):
        pass

    def estimate(self, t: np.ndarray):
        pass
