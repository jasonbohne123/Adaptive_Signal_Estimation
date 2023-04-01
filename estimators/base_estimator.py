import abc
from collections import defaultdict

import numpy as np


class Base_Estimator(abc.ABC):
    """
    Abstract Base Class for all estimators; which have a fit and estimate method
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

    @abc.abstractmethod
    def fit(self, warm_start=False):
        """Fit estimator to data given hyperparameters"""

    @abc.abstractmethod
    def estimate(self, t: np.ndarray):
        """Estimate y_hat given x and hyperparameters"""

    @abc.abstractmethod
    def update_params(self, hypers: dict):
        """Update parameters of estimator"""
        self.hypers.update(hypers)
