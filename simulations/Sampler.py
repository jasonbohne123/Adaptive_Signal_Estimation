import numpy as np

from simulations.Simulator import Simulator


class Sampler(Simulator):
    """Base Sampler Class for a given distribution and true process"""

    def __init__(self, distribution, rng=None):
        self.distribution = distribution
        self.underlying_simulator = Simulator(distribution, rng)
        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

    def sample(self, true_processes, n_samples=None, scale=None):
        """Generate samples from the distribution of true_processes"""

        # if no scale is given, use the standard deviation of the true process
        if scale is None:
            scale = np.std(true_processes)

        if n_samples == 1:
            n_samples = true_processes.shape

        return self.underlying_simulator.simulate(n=n_samples, loc=true_processes, scale=scale, rng=self.rng)
