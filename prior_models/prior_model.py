import abc


class Prior(abc.ABC):
    """Abstract class for prior models

    Models for prior are one-dimensional and deterministic
    """

    def __init__(self, prior, t=None):
        self.prior = prior
        self.t = t
        self.time_flag = False if t is None else True
        self.name = self.__class__.__name__

    @abc.abstractmethod
    def get_prior(self):
        return self.prior

    def get_name(self):
        return self.name

    def get_time_flag(self):
        return self.time_flag
