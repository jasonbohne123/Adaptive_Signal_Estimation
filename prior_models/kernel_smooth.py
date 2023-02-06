from kernel_methods.Kernel_Smoother import KernelSmoother
from prior_models.prior_model import Prior


class Kernel_Smooth_Prior(Prior):
    def __init__(self, model: Prior):

        # store original model
        self.orig_model = model

        assert model.t is not None, "Time series must be provided for kernel smoothing"

        # initialize the prior model off real data
        prior = KernelSmoother(model.t, model.prior, bandwidth_style=0)

        # perform the kernel smoothing
        prior = prior.smooth_series(prior.fit())

        # update the prior
        super().__init__(prior, model.t)
        self.name = "Kernel_Smooth_Prior"
        self.prior = prior
        self.orig_data = model.prior
        self.t = [0.01 * i for i in model.t]  # scaling might be required here
        self.time_flag = model.time_flag

        self.adaptive_penalty = True
        self.estimator = True

    def get_prior(self):

        return self.prior

    def get_name(self):
        return self.name

    def get_time_flag(self):
        return self.time_flag
