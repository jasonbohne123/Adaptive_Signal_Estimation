import sys

sys.path.append("../")
import itertools
import random
import string
import time
from typing import Dict

import numpy as np

from prior_models.deterministic_prior import Deterministic_Prior
from prior_models.kernel_smooth import Kernel_Smooth_Prior
from prior_models.prior_model import Prior
from prior_models.volume_prior import Volume_Prior
from simulations.generate_sims import generate_samples, generate_true_dgp
from trend_filtering.test_adaptive_tf import test_adaptive_tf
from trend_filtering.tf_constants import get_model_constants, get_simulation_constants


def run_bulk_trend_filtering(
    exp_name: str,
    true: np.ndarray,
    true_knots: np.ndarray,
    samples: np.ndarray,
    prior_model: Prior,
    sim_grid: Dict = None,
    verbose=True,
):
    """Solves Bulk Trend Filtering problems

    m: Number of simulations
    n: Length of each path

    """
    start_time = time.time()

    # decompose prior into raw data
    submodel = prior_model.submodel

    # # set prior to interior cp
    # submodel.prior[true_knots] = 1
    # submodel.prior[np.setdiff1d(np.arange(len(submodel.prior)), true_knots)] = 0.1

    # biased prior to true cp
    updated_prior = Deterministic_Prior(np.ones(len(submodel.t)), submodel.t)

    print(submodel.t)

    # # smooth around indicator
    kernel_smooth_prior = Kernel_Smooth_Prior(updated_prior, sim_grid["bandwidth"])

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
        t=submodel.t,
        sim_grid=sim_grid,
        prev_results=None,
    )

    # adaptive penalty
    new_results = apply_function_to_paths(
        samples,
        test_adaptive_tf,
        exp_name=exp_name,
        prior_model=kernel_smooth_prior,
        t=submodel.t,
        flags=flags,
        true=true,
        true_knots=true_knots,
        sim_grid=sim_grid,
        prev_results=results,
    )

    total_time = time.time() - start_time
    if verbose:
        print(f"Total time taken: is {round(total_time,2)} for {n_sims} simulations ")

    return


def apply_function_to_paths(paths, function, prior_model, t, exp_name, flags, true, true_knots, sim_grid, prev_results):
    """Apply a function to each path in a set of simulations"""
    results = {}

    for i, sample_path in enumerate(paths):

        if prev_results is not None:
            # pass in any grid parameters and previous results if applicable
            args = {"bandwidth": sim_grid["bandwidth"], "snr": sim_grid["snr"], "non_adaptive_results": prev_results[i]}

        else:
            # pass in any grid parameters if applicable
            args = {"bandwidth": sim_grid["bandwidth"], "snr": sim_grid["snr"], "non_adaptive_results": None}

        results[i] = function(
            sample_path,
            exp_name=exp_name,
            flags=flags,
            true_sol=true[i],
            true_knots=true_knots,
            prior_model=prior_model,
            t=t,
            args=args,
        )

    return results


# python run_bulk_tf.py
if __name__ == "__main__":

    random_letters = "".join(random.choice(string.ascii_uppercase) for i in range(5))
    exp_name = f"L1_Trend_Filter_{random_letters}"

    n = get_model_constants().get("n")
    n_sims = get_simulation_constants().get("n_sims")

    # simulation priors
    # prior_model = Normal_Prior(n, time_flag=True)
    # prior_model = Uniform_Prior(n)

    # real data prior
    prior_model = Kernel_Smooth_Prior(Volume_Prior(n, time_flag=True))

    # simulation style

    sim_style = "piecewise_linear" if get_model_constants().get("order") == 1 else "piecewise_constant"

    # generate true dgp off optimal kde prior
    true, true_knots, cp_knots = generate_true_dgp(prior_model, sim_style)

    # evaluation for new sims with different bandwidths
    bandwidth_grid = [50]
    snr_grid = [0.05]

    possible_comb = itertools.product(bandwidth_grid, snr_grid)

    print("Running {n_sims} simulations of length {n}".format(n_sims=n_sims, n=n))
    print("Experiment is {sim_style} ".format(sim_style=sim_style))
    print("Experiment is Time Aware of {time_flag} ".format(time_flag=prior_model.time_flag))

    for pair in possible_comb:

        adjusted_true, samples = generate_samples(true, true_knots, pair[1])

        sim_grid = {"bandwidth": pair[0], "snr": pair[1]}
        run_bulk_trend_filtering(exp_name, adjusted_true, true_knots, samples, prior_model, sim_grid=sim_grid)
