import numpy as np

from simulations.Conditional_Simulator import ConditionalSimulator
from simulations.Sampler import Sampler


def generate_conditional_piecewise_paths(
    prior, sim_style, label_style="k_maxima", k_points=10, underling_dist="normal", n_sims=10000, scale=10e-3
):
    """Generate piecewise constant/linear paths with changepoints at the k_maxima of the prior distribution"""

    sim = ConditionalSimulator(prior, sim_style, label_style, k_points, underling_dist, n_sims=n_sims)
    true = sim.simulate()

    true=sim.evaluate_within_sample(sim.cp_index, true)

    sampler=Sampler(underling_dist)

    samples=sampler.sample(true,scale=scale)

    return samples


def apply_function_to_paths(paths, function,exp_name,flags):
    """Apply a function to each path in a set of simulations"""

    return np.apply_along_axis(function, 1, paths,exp_name=exp_name,flags=flags)


