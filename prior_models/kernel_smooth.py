from kernel_methods.Kernel_Smoother import KernelSmoother
from prior_models.prior_model import Prior


class Kernel_Smooth_Prior(Prior):
    """This class is used to create a prior model from a kernel smoother"""

    def __init__(self, model: Prior, preselected_bandwidth=None):

        # initialize the prior model off real data, with optional preselected bandwidth
        prior_model = KernelSmoother(
            model.t, model.prior, bandwidth_style=0, preselected_bandwidth=preselected_bandwidth
        )

        # perform the kernel smoothing on the prior (here is log volume)
        prior_series = prior_model.smooth_series(prior_model.fit())

        # update the prior from defaults
        super().__init__(prior_series, model.t)
        self.name = "Kernel_Smooth_Prior"
        self.prior = prior_series
        self.bandwidth = prior_model.optimal_bandwidth
        self.estimator = True
        self.orig_data = model.orig_data

        # store the submodel (properties for estimator)
        self.submodel = model

    def get_prior(self):

        return self.prior

    def get_name(self):
        return self.name

    def get_time_flag(self):
        return self.time_flag
