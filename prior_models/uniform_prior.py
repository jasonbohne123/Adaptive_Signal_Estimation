import numpy as np

from prior_models.prior_model import Prior


class UniformPrior(Prior):
    """Prior Model sampling from a uniform distribution"""

    def __init__(self, n, lb=0.5, ub=2, t=None):
        prior = np.random.uniform(lb, ub, n)

        # update prior from defaults
        super().__init__(prior, t)
        self.name = "Uniform_Prior"
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
