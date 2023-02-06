import time
from typing import Dict, Union

import numpy as np

from evaluation_metrics.loss_functions import compute_error
from prior_models.prior_model import Prior
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.helpers import log_to_mlflow, perform_cv, prep_signal, write_to_files
from trend_filtering.tf_constants import get_model_constants


def test_adaptive_tf(
    sample: np.ndarray,
    t: Union[None, np.ndarray] = None,
    true_sol: Union[None, np.ndarray] = None,
    true_knots: Union[None, np.ndarray] = None,
    prior_model: Union[Prior, None] = None,
    exp_name="DEFAULT",
    flags: Dict[str, bool] = None,
):
    """Wrapper function to apply trend filtering to a single path

    For Adaptive TF, the prior is a Prior Model object. For Constant TF, the prior is None
    
    """

    start_time = time.time()
    include_cv, plot, verbose, bulk, log_mlflow = map(
        flags.get, ["include_cv", "plot", "verbose", "bulk", "log_mlflow"]
    )

    sample, true_sol, D, time_flag = prep_signal(sample, true_sol, t)

    if not include_cv and not prior:
        print(" No prior provided and no cross validation")
        return

    # prep numerial prior if adaptive
    if prior_model:
        prior = prior_model.prior[1:-1]

    else:
        prior = None

    # perform cross validation if flagged
    if include_cv:
        prior, best_scaler = perform_cv(sample, D, prior, t)

    # reconstruct signal
    results = adaptive_tf(sample, D_=D, t=t, prior=prior, select_knots=get_model_constants()["solve_cp"])
    results["computation_time"] = time.time() - start_time

    if verbose:
        print(f"Solved TF problem with status: {results['status']}")

        max_iter = get_model_constants()["maxiter"]
        print(f" Total Iterations: {results['iters']} out of {max_iter}")

    # extract solution information
    sol = results["sol"]
    if sol is not None:
        sol_array = sol.x
        knots = sol.knots if get_model_constants()["solve_cp"] else None
    else:
        print("No solution found")
        return

    # compute mse from sample and true
    mse_from_sample = compute_error(sample, sol_array, type="mse")
    mse_from_true = compute_error(true_sol, sol_array, type="mse")

    expected_prediction_error = compute_error(true_sol, sol_array, type="epe")

    if get_model_constants()["solve_cp"] and knots is not None:
        hausdorff_distance = compute_error(knots, true_knots, type="hausdorff")

        if verbose:
            print(f"True knots: {true_knots}")
            print(f"Estimated knots: {knots}")
            print(f"Hausdorff distance: {hausdorff_distance}")

    if verbose:
        print(" ")

    # write artifacts to files
    write_to_files(sample, true_sol, sol_array, prior_model, true_knots, knots, plot)

    # log information to mlflow
    if log_mlflow:
        log_to_mlflow(
            exp_name,
            results,
            prior_model,
            best_scaler,
            mse_from_sample,
            mse_from_true,
            expected_prediction_error,
            hausdorff_distance,
            len(true_knots),
            len(knots),
            flags,
        )

    return
