import abc


class Prior(abc.ABC):
    """Abstract class for prior models

    Models for prior are one-dimensional and deterministic
    """

    def __init__(self, prior, t=None):

        # set prior and original data
        self.prior = prior
        self.orig_data = prior

        # account for time
        self.t = t

        # flags
        self.time_flag = False
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
