from prior_models.prior_model import Prior


class Deterministic_Prior(Prior):
    """Deterministic Prior Model"""

    def __init__(self, prior,t):
 
        # update prior from defaults
        super().__init__(prior, t)
        self.name = "Deterministic_Prior"
        self.prior = prior
        self.time_flag = True
        self.t = t


    def get_prior(self):
        return self.prior

    def get_name(self):
        return self.name

    def get_time_flag(self):
        return self.time_flag