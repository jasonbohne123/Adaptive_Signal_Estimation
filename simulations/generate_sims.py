from simulations.Conditional_Simulator import ConditionalSimulator
from simulations.Sampler import Sampler
from trend_filtering.tf_constants import get_simulation_constants


def generate_conditional_piecewise_paths(prior, sim_style, label_style="k_maxima"):
    """Generate piecewise constant/linear paths with changepoints at the k_maxima of the prior distribution"""

    # fetch simulation constants from prespecified file (tf_constants.py)
    underlying_dist, signal_to_noise, reference_variance, n_samples = map(
        get_simulation_constants().get, ["underlying_dist", "signal_to_noise", "reference_variance", "n_samples"]
    )
    sim = ConditionalSimulator(prior, sim_style)
    true = sim.simulate()
    true_knots = sim.interior_index

    # (n_sims,len_sims)
    true = sim.evaluate_within_sample(sim.cp_index, true)

    sampler = Sampler(underlying_dist)

    # (n_sims,len_sims), (n_sims*n_samples,len_sims)
    samples, adjusted_true = sampler.sample(true, n_samples=n_samples, scale=reference_variance / signal_to_noise)

    return adjusted_true, samples, true_knots
