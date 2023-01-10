import sys

sys.path.append("../")
import random
import string
import time

import numpy as np

from simulations.generate_sims import apply_function_to_paths, generate_conditional_piecewise_paths
from trend_filtering.test_adaptive_tf import test_adaptive_tf


def run_bulk_trend_filtering(prior, sim_style, m=500, n=100, verbose=True):
    """Solves Bulk Trend Filtering problems

    m: Number of simulations
    n: Length of each path

    """
    time.time()
    prior = prior[:m]

    # generate samples
    samples = generate_conditional_piecewise_paths(prior, sim_style, n_sims=n)

    random_letters = "".join(random.choice(string.ascii_uppercase) for i in range(5))
    exp_name = f"L1_Trend_Filter_{random_letters}"

    if verbose:
        print("Running {m} simulations of length {n}".format(n=m, m=n))
        print("Experiment is {sim_style} ".format(sim_style=sim_style))

    # apply tf to each path with specified flags
    flags = {"include_cv": True, "plot": True, "verbose": True, "bulk": True, "log_mlflow": True}
    results = apply_function_to_paths(samples, test_adaptive_tf, exp_name=exp_name, flags=flags)

    time.time()
    if verbose:
        print(f"Total time taken: is {round(time,2)} for {n} simulations ")

    return


# python run_bulk_tf.py
if __name__ == "__main__":
    prior = np.random.normal(size=500)
    sim_style = "piecewise_linear"
    run_bulk_trend_filtering(prior, sim_style)
