import numpy as np

from prior_models.prior_model import Prior


class Uniform_Prior(Prior):
    """Prior Model sampling from a uniform distribution"""

    def __init__(self, n, lb=0.5, ub=2):
        prior = np.random.uniform(lb, ub, n)

        # update prior from defaults
        super().__init__(prior, t)
        self.name = "Uniform_Prior"
        self.prior = prior

    def get_prior(self):
        return self.prior

    def get_name(self):
        return self.name
