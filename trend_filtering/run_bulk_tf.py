import sys

sys.path.append("../")
import random
import string
import time

import numpy as np

from simulations.generate_sims import apply_function_to_paths, generate_conditional_piecewise_paths
from trend_filtering.tf_constants import get_simulation_constants, get_model_constants
from trend_filtering.test_adaptive_tf import test_adaptive_tf


def run_bulk_trend_filtering(prior, sim_style,n_sims,n, verbose=True):
    """Solves Bulk Trend Filtering problems

    m: Number of simulations
    n: Length of each path

    """
    start_time = time.time()
    prior = prior[:n]

    # generate samples
    true, samples = generate_conditional_piecewise_paths(prior, sim_style)

    random_letters = "".join(random.choice(string.ascii_uppercase) for i in range(5))
    exp_name = f"L1_Trend_Filter_{random_letters}"

    if verbose:
        print("Running {n_sims} simulations of length {n}".format(n_sims=n_sims, n=n))
        print("Experiment is {sim_style} ".format(sim_style=sim_style))

    # apply tf to each path with specified flags
    flags = {"include_cv": True, "plot": True, "verbose": True, "bulk": True, "log_mlflow": True}

    # constant penalty
    results = apply_function_to_paths(samples, test_adaptive_tf, exp_name=exp_name, flags=flags, true=true,lambda_p=None)

    unpadded_prior=prior[1:-1]
    # adaptive penalty
    new_results = apply_function_to_paths(
        samples, test_adaptive_tf, exp_name=exp_name, lambda_p=unpadded_prior, flags=flags, true=true
    )

    total_time = time.time() - start_time
    if verbose:
        print(f"Total time taken: is {round(total_time,2)} for {n_sims} simulations ")

    return


# python run_bulk_tf.py
if __name__ == "__main__":
    n=get_model_constants().get("n")
    n_sims=get_simulation_constants().get("n_sims")

    prior = np.random.uniform(0.5,2, n)
    sim_style = "piecewise_linear"
    
    run_bulk_trend_filtering(prior, sim_style,n_sims,n)
