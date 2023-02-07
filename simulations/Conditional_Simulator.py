import numpy as np

from trend_filtering.label_changepoints import label_changepoints
from trend_filtering.tf_constants import get_simulation_constants

from .Simulator import Simulator

# TODO: add timestep functionality in simulation


class ConditionalSimulator(Simulator):
    def __init__(
        self,
        prior,
        sim_style,
        rng=None,
    ):
        self.prior = prior
        self.sim_style = sim_style

        # fetch simulation constants from prespecified file (tf_constants.py)
        self.underlying_dist, self.variance_scaling, self.k_points, self.label_style, self.n_sims, self.shift = map(
            get_simulation_constants().get,
            ["underlying_dist", "reference_variance", "k_points", "label_style", "n_sims", "shift"],
        )

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.underlying_simulator = Simulator(self.underlying_dist, self.rng, self.variance_scaling)
        self.n_sims = self.n_sims

        self.cp_index = label_changepoints(prior, self.label_style, self.k_points)

        # index of interior points (knots)
        self.interior_index = np.setdiff1d(self.cp_index, np.array([0, len(prior) - 1]))

    def simulate(self):
        """Sample from the prior distribution at cp_index"""

        sampled_cp = np.zeros((self.n_sims, len(self.prior)))

        sampled_cp[:, 0] = np.zeros(self.n_sims)

        sampled_cp[:, self.cp_index] = self.underlying_simulator.simulate(
            n=(self.n_sims, len(self.cp_index)), rng=self.rng
        )

        return sampled_cp

    def evaluate_within_sample(self, cp_index, sampled_cp):
        """Generate known points (from style) within sampled index"""

        true_processes = np.zeros((self.n_sims, len(self.prior)))

        # process is piecewise constant
        if self.sim_style == "piecewise_constant":

            for i in range(len(cp_index) - 1):
                # evaluate remaining of the interval to be constant
                true_processes[:, cp_index[i] : cp_index[i + 1]] = np.multiply(
                    sampled_cp[:, cp_index[i]].reshape(-1, 1), np.ones((self.n_sims, cp_index[i + 1] - cp_index[i]))
                )

        # process is piecewise linear
        elif self.sim_style == "piecewise_linear":
            for i in range(len(cp_index) - 1):

                steps = cp_index[i + 1] - cp_index[i]

                if i == 0:
                    # sample for second degree of freedom
                    true_processes[:, 1] = self.underlying_simulator.simulate(n=self.n_sims, rng=self.rng)
                    diff = sampled_cp[:, 0]
                else:
                    diff = sampled_cp[:, cp_index[i]]

                # vectorizing and writing in terms of indices subsequent to cp
                end_point = true_processes[:, cp_index[i]] + steps * diff

                # update simulated value at cp_index[i+1]
                true_processes[:, cp_index[i] + 1 : cp_index[i + 1] + 1] = np.linspace(
                    true_processes[:, cp_index[i]] + sampled_cp[:, cp_index[i]],
                    end_point,
                    cp_index[i + 1] - cp_index[i],
                ).T
        else:
            raise ValueError("Invalid simulation style")

        return true_processes + self.shift
