from prior_models.prior_model import Prior


class Deterministic_Prior(Prior):
    """Deterministic Prior Model"""

    def __init__(self, prior):

        # update prior from defaults
        self.name = "Deterministic_Prior"
        self.prior = prior

    def get_prior(self):
        return self.prior

    def get_name(self):
        return self.name
