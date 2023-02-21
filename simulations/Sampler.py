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

        # if no number of samples is given, use the number of true processes
        if n_samples is None:
            n_samples = 1

        # this requires us to repeat the true processes n_samples times (t_1, t_2, ...t_n) repeated n_samples times
        adjusted_true = np.repeat(true_processes, n_samples, axis=0)

        # (n_sims,len_sims), (n_sims*n_samples,len_sims)
        # each sample centered on true with scale across all dimensions
        return (
            self.underlying_simulator.simulate(
                (n_samples * true_processes.shape[0], true_processes.shape[1]),
                loc=adjusted_true,
                scale=scale,
                rng=self.rng,
            ),
            adjusted_true,
        )
