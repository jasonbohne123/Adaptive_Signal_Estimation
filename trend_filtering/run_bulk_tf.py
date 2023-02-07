import sys

sys.path.append("../")
import random
import string
import time

from prior_models.kernel_smooth import Kernel_Smooth_Prior
from prior_models.prior_model import Prior
from prior_models.volume_prior import Volume_Prior
from prior_models.uniform_prior import Uniform_Prior
from simulations.generate_sims import generate_conditional_piecewise_paths
from trend_filtering.test_adaptive_tf import test_adaptive_tf
from trend_filtering.tf_constants import get_model_constants, get_simulation_constants


def run_bulk_trend_filtering(prior_model: Prior, sim_style: str, n_sims: int, n: int, verbose=True):
    """Solves Bulk Trend Filtering problems

    m: Number of simulations
    n: Length of each path

    """
    start_time = time.time()

    # generate samples
    # this uses local maximas to generate the paths on the smooth prior
    true, samples, true_knots = generate_conditional_piecewise_paths(prior_model.prior, sim_style)

    random_letters = "".join(random.choice(string.ascii_uppercase) for i in range(5))
    exp_name = f"L1_Trend_Filter_{random_letters}"

    if verbose:
        print("Running {n_sims} simulations of length {n}".format(n_sims=n_sims, n=n))
        print("Experiment is {sim_style} ".format(sim_style=sim_style))

    # apply tf to each path with specified flags
    flags = {"include_cv": True, "plot": True, "verbose": True, "bulk": True, "log_mlflow": True}

    # constant penalty
    results = apply_function_to_paths(
        samples,
        test_adaptive_tf,
        exp_name=exp_name,
        flags=flags,
        true=true,
        true_knots=true_knots,
        prior_model=None,
        t=None,
    )

    # adaptive penalty
    new_results = apply_function_to_paths(
        samples,
        test_adaptive_tf,
        exp_name=exp_name,
        prior_model=prior_model,
        t=None,
        flags=flags,
        true=true,
        true_knots=true_knots,
    )

    total_time = time.time() - start_time
    if verbose:
        print(f"Total time taken: is {round(total_time,2)} for {n_sims} simulations ")

    return


def apply_function_to_paths(paths, function, prior_model, t, exp_name, flags, true, true_knots):
    """Apply a function to each path in a set of simulations"""

    for i, sample_path in enumerate(paths):
        function(
            sample_path,
            exp_name=exp_name,
            flags=flags,
            true_sol=true[i],
            true_knots=true_knots,
            prior_model=prior_model,
            t=t,
        )

    return


# python run_bulk_tf.py
if __name__ == "__main__":
    n = get_model_constants().get("n")
    n_sims = get_simulation_constants().get("n_sims")

    # simulation priors
    # prior_model = Normal_Prior(n, time_flag=True)
    #prior_model = Uniform_Prior(n)

    # real data prior
    prior_model = Kernel_Smooth_Prior(Volume_Prior(n, time_flag=False))

    sim_style = "piecewise_linear" if get_model_constants().get("order") == 1 else "piecewise_constant"

    run_bulk_trend_filtering(prior_model, sim_style, n_sims, n)
