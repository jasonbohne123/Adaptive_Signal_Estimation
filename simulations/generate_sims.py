from simulations.Conditional_Simulator import ConditionalSimulator
from simulations.Sampler import Sampler
from trend_filtering.tf_constants import get_trend_filtering_simulation_constants


def generate_conditional_piecewise_paths(prior, sim_style, label_style="k_maxima"):
    """Generate piecewise constant/linear paths with changepoints at the k_maxima of the prior distribution"""
    k_points, underlying_dist, n_sims, sample_variance = map(
        get_trend_filtering_simulation_constants().get, ["k_points", "underlying_dist", "n_sims", "sample_variance"]
    )

    sim = ConditionalSimulator(prior, sim_style, label_style, k_points, underlying_dist, n_sims=n_sims)
    true = sim.simulate()

    true = sim.evaluate_within_sample(sim.cp_index, true)

    sampler = Sampler(underlying_dist)

    samples = sampler.sample(true, scale=sample_variance)

    return true, samples


def apply_function_to_paths(paths, function, exp_name, flags, true):
    """Apply a function to each path in a set of simulations"""

    for i, sample_path in enumerate(paths):
        function(sample_path, exp_name=exp_name, flags=flags, true_sol=true[i])

    return
