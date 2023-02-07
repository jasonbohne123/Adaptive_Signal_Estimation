import numpy as np

from prior_models.prior_model import Prior


class Normal_Prior(Prior):
    """Prior Model sampling from a normal distribution"""

    def __init__(self, n, mu=1, sigma=0.25, t=None):
        prior = np.random.normal(mu, sigma, n)

        # update prior from defaults
        super().__init__(prior, t)
        self.name = "Normal_Prior"
        self.prior = prior

        if t is not None:
            self.time_flag = True
            self.t = t

    def get_prior(self):
        return self.prior

    def get_name(self):
        return self.name

    def get_time_flag(self):
        return self.time_flag
