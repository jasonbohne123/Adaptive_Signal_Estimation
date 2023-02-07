from prior_models.prior_model import Prior


class Deterministic_Prior(Prior):
    """Deterministic Prior Model"""

    def __init__(self, prior,t=None):

        if t is not None:
            self.time_flag = True
            self.t = t
            super().__init__(prior, t)
        
        else:
            self.time_flag = False
            super().__init__(prior)
 
        # update prior from defaults
        self.name = "Deterministic_Prior"
        self.prior = prior
 

    def get_prior(self):
        return self.prior

    def get_name(self):
        return self.name

    def get_time_flag(self):
        return self.time_flag