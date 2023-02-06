from kernel_methods.Kernel_Smoother import KernelSmoother
from prior_models.prior_model import Prior


class Kernel_Smooth_Prior(Prior):
    """ This class is used to create a prior model from a kernel smoother"""
    def __init__(self, model: Prior):

        # initialize the prior model off real data
        prior = KernelSmoother(model.t, 1 / model.prior, bandwidth_style=0)

        # perform the kernel smoothing
        prior = prior.smooth_series(prior.fit())

        # update the prior from defaults
        super().__init__(prior, model.t)
        self.name = "Kernel_Smooth_Prior"
        self.prior = prior
        self.estimator = True
        self.orig_data = model.orig_data

    def get_prior(self):

        return self.prior

    def get_name(self):
        return self.name

    def get_time_flag(self):
        return self.time_flag
