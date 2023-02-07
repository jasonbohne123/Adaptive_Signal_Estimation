import abc
import numpy as np

class Prior(abc.ABC):
    """Abstract class for prior models

    Models for prior are one-dimensional and deterministic
    """

    def __init__(self, prior, t=None):

        # set prior and original data
        self.prior = prior
        self.orig_data = prior

        # account for time
        if t is not None:
            self.time_flag = True
            self.t = t

        else:
            self.time_flag = False
            self.t=np.arange(len(prior)) # uniform time

        # flags
        self.adaptive_penalty = True
        self.estimator = False

        self.name = self.__class__.__name__

    @abc.abstractmethod
    def get_prior(self):
        return self.prior

    @abc.abstractmethod
    def get_name(self):
        return self.name

    @abc.abstractmethod
    def get_time_flag(self):
        return self.time_flag
