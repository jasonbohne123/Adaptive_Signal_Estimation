from simulations.Conditional_Simulator import ConditionalSimulator
from simulations.Sampler import Sampler
from trend_filtering.tf_constants import get_simulation_constants


def generate_conditional_piecewise_paths(prior, sim_style, label_style="k_maxima"):
    """Generate piecewise constant/linear paths with changepoints at the k_maxima of the prior distribution"""

    # fetch simulation constants from prespecified file (tf_constants.py)
    underlying_dist, signal_to_noise, reference_variance = map(
        get_simulation_constants().get, ["underlying_dist", "signal_to_noise", "reference_variance"]
    )
    sim = ConditionalSimulator(prior, sim_style)
    true = sim.simulate()

    # (n_sims,len_sims)
    true = sim.evaluate_within_sample(sim.cp_index, true)

    sampler = Sampler(underlying_dist)

    # (n_sims,len_sims)
    samples = sampler.sample(true, scale=reference_variance * signal_to_noise)

    return true, samples


def apply_function_to_paths(paths, function, exp_name, flags, true, lambda_p=None):
    """Apply a function to each path in a set of simulations"""

    for i, sample_path in enumerate(paths):
        function(sample_path, exp_name=exp_name, flags=flags, true_sol=true[i], lambda_p=lambda_p)

    return
