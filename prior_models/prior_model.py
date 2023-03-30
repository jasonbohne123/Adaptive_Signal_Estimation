import abc


class Prior(abc.ABC):
    """Abstract class for prior models

    prior: np.ndarray (n)
    """

    def __init__(self, prior):

        # set prior and original data
        self.prior = prior
        self.name = self.__class__.__name__

    @abc.abstractmethod
    def get_prior(self):
        return self.prior

    @abc.abstractmethod
    def get_name(self):
        return self.name
