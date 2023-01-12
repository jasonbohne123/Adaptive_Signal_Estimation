import math

import numpy as np

from trend_filtering.label_changepoints import label_changepoints

from .Simulator import Simulator

# TODO: add timestep functionality in simulation


class ConditionalSimulator(Simulator):
    def __init__(
        self,
        prior,
        sim_style,
        label_style,
        k_points,
        underlying_dist=None,
        n_sims=1000,
        variance_scaling=10e-3,
        rng=None,
        shift=100,
    ):
        self.prior = prior
        self.sim_style = sim_style
        self.label_style = label_style
        self.shift = shift

        if k_points > len(prior):
            print("k_points is greater than the length of the prior. Setting k_points to half of length of the prior")
            k_points = math.floor(len(prior) / 2)

        self.k_points = k_points

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        self.underlying_simulator = Simulator(underlying_dist, self.rng, variance_scaling)
        self.n_sims = n_sims

        self.cp_index = label_changepoints(prior, label_style, k_points)

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
