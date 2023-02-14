import sys

from simulations.Conditional_Simulator import ConditionalSimulator
from simulations.Sampler import Sampler
from trend_filtering.tf_constants import get_simulation_constants

sys.path.append("../")

from prior_models.prior_model import Prior


def generate_true_dgp(prior: Prior, sim_style, label_style="k_local_spikes"):

    sim = ConditionalSimulator(prior.prior, sim_style)
    true = sim.simulate()
    true_knots = sim.interior_index
    cp_knots = sim.cp_index

    # (n_sims,len_sims)
    true = sim.evaluate_within_sample(cp_knots, true)

    return true, true_knots, cp_knots


def generate_samples(true, true_knots, snr=None):
    """Generate piecewise constant/linear paths with changepoints at the k_maxima of the prior distribution"""

    # fetch simulation constants from prespecified file (tf_constants.py)
    underlying_dist, reference_variance, n_samples = map(
        get_simulation_constants().get, ["underlying_dist", "reference_variance", "n_samples"]
    )

    # get default snr if not specified
    if snr is None:
        snr = get_simulation_constants().get("signal_to_noise")

    sampler = Sampler(underlying_dist)

    # (n_sims,len_sims), (n_sims*n_samples,len_sims)
    samples, adjusted_true = sampler.sample(true, n_samples=n_samples, scale=reference_variance / snr)

    return adjusted_true, samples
