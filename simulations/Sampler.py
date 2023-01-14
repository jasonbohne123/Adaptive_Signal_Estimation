import numpy as np

from simulations.Simulator import Simulator
from trend_filtering.tf_constants import get_simulation_constants


class Sampler(Simulator):
    """Base Sampler Class for a given distribution and true process"""

    def __init__(self, distribution, rng=None, variance_scaling=None):
        """Initialize the sampler with a distribution and a random number generator"""

        self.distribution = distribution
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        if variance_scaling is None:
            self.variance_scaling = get_simulation_constants().get("reference_variance")
        else:
            self.variance_scaling = variance_scaling

        self.underlying_simulator = Simulator(distribution, rng, self.variance_scaling)

    def sample(self, true_processes, n_samples=None, scale=None):
        """Generate samples from the distribution of true_processes"""

        # if no scale is given, use the standard deviation of the true process
        if scale is None:
            scale = np.std(true_processes)

        if n_samples == 1:
            n_samples = true_processes.shape

        return self.underlying_simulator.simulate(n=n_samples, loc=true_processes, scale=scale, rng=self.rng)
