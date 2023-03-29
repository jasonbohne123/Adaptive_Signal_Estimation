from abc import ABC
from collections import defaultdict

import numpy as np


class Base_Estimator(ABC):
    """
    Base class for all estimators; which have a fit and estimate method
    """

    def __init__(self, x, y):
        self.name = None
        self.x = x
        self.y = y

        self.y_hat = None

        # hyperparameters and respective max values ([0,max])
        self.hypers = defaultdict(float)
        self.hyper_max = defaultdict(float)

        # configs for model
        self.configs = dict()

    def fit(self, warm_start=False):
        """Fit estimator to data given hyperparameters"""

        pass

    def estimate(self, t: np.ndarray):
        """Estimate y_hat given x and hyperparameters"""
        pass

    def update_params(self, hypers: dict):
        """Update parameters of estimator"""
        self.hypers.update(hypers)

        pass
