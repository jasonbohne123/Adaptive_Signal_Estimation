import time
from typing import Dict, Union

import numpy as np
from trend_filtering.cv_tf import perform_cv
from trend_filtering.primal_dual_tf import adaptive_tf
from trend_filtering.test_helpers import log_to_mlflow, prep_signal, write_to_files

from estimators.trend_filtering.helpers.primal_dual_constants import get_model_constants
from evaluation_metrics.loss_functions import compute_error
from prior_models.prior_model import Prior


def test_adaptive_tf(
    sample: np.ndarray,
    t: Union[None, np.ndarray] = None,
    true_sol: Union[None, np.ndarray] = None,
    true_knots: Union[None, np.ndarray] = None,
    prior_model: Union[Prior, None] = None,
    exp_name: str = "DEFAULT",
    args: Dict[str, Union[str, float, np.ndarray]] = None,
    flags: Dict[str, bool] = None,
):
    """Wrapper function to apply trend filtering to a single path

    For Adaptive TF, the prior is a Prior Model object. For Constant TF, the prior is None

    """

    start_time = time.time()
    include_cv, plot, verbose, bulk, log_mlflow = map(
        flags.get, ["include_cv", "plot", "verbose", "bulk", "log_mlflow"]
    )

    sample, true_sol, D = prep_signal(sample, true_sol, prior_model=prior_model, t=t)

    time_aware = D.time_enabled
    adaptive_tf_flag = D.prior_enabled

    flags["time_aware"] = time_aware
    flags["adaptive_tf"] = adaptive_tf_flag

    best_scaler = perform_cv(sample, D)

    # reconstruct signal (allows for time)
    results = adaptive_tf(
        sample, D_=D, lambda_p=best_scaler, select_knots=get_model_constants()["solve_cp"], true_knots=true_knots
    )
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
        spline = sol.fit_linear_spline()

    else:
        print("No solution found")
        return

    # compute mse from sample and true
    mse_from_sample = compute_error(sample, sol_array, type="mse")
    mse_from_true = compute_error(true_sol, sol_array, type="mse")
    spline_mse = compute_error(true_sol, spline, type="mse")

    expected_prediction_error = compute_error(true_sol, sol_array, type="epe")

    if get_model_constants()["solve_cp"] and knots is not None:
        hausdorff_distance = compute_error(knots, true_knots, type="hausdorff")

        if verbose:
            print(f"True knots: {true_knots}")
            print(f"Estimated knots: {knots}")
            print(f"Hausdorff distance: {hausdorff_distance}")

    print(" ")

    # write artifacts to files
    write_to_files(sample, true_sol, sol_array, spline, prior_model, true_knots, knots, plot)

    # log information to mlflow
    if log_mlflow:
        adaptive_results = {
            "mse_from_sample": mse_from_sample,
            "mse_from_true": mse_from_true,
            "spline_mse": spline_mse,
            "hausdorff_distance": hausdorff_distance,
            "expected_prediction_error": expected_prediction_error,
        }

        log_to_mlflow(
            exp_name,
            results,
            prior_model,
            best_scaler,
            adaptive_results,
            args["non_adaptive_results"],
            args["snr"],
            len(true_knots),
            len(knots),
            flags,
        )

    return {"mse_from_true": mse_from_true, "spline_mse": spline_mse, "hausdorff_distance": hausdorff_distance}