import time
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np

from evaluation_metrics.loss_functions import compute_error
from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from simulations.mlflow_helpers import create_mlflow_experiment, log_mlflow_params
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.cv_tf import cross_validation
from trend_filtering.tf_constants import get_model_constants, get_simulation_constants


def test_adaptive_tf(
    sample: np.ndarray,
    t: Union[None, np.ndarray] = None,
    true_sol: Union[None, np.ndarray] = None,
    true_knots: Union[None, np.ndarray] = None,
    lambda_p: Union[np.ndarray, None] = None,
    exp_name="DEFAULT",
    flags: Dict[str, bool] = None,
):
    """Test adaptive_tf function"""

    start_time = time.time()
    include_cv, plot, verbose, bulk, log_mlflow = map(
        flags.get, ["include_cv", "plot", "verbose", "bulk", "log_mlflow"]
    )

    sample, true_sol, D, time_flag = prep_signal(sample, true_sol, t)

    if not include_cv and not lambda_p:
        print(" No lambda_p provided and no cross validation")
        return

    # perform cross validation if flagged
    if include_cv:
        lambda_p, best_lambda = perform_cv(sample, D, lambda_p, t)

    # reconstruct signal
    results = adaptive_tf(sample, D_=D, t=t, lambda_p=lambda_p, select_knots=get_model_constants()["solve_cp"])
    results["computation_time"] = time.time() - start_time

    if verbose:
        print(f"Solved TF problem with status: {results['status']}")

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
    write_to_files(sample, true_sol, sol_array, true_knots, knots, plot, lambda_p)

    # log information to mlflow
    if log_mlflow:
        log_to_mlflow(
            exp_name,
            results,
            lambda_p,
            best_lambda,
            mse_from_sample,
            mse_from_true,
            expected_prediction_error,
            hausdorff_distance,
            len(true_knots),
            len(knots),
            flags,
        )

    return


def prep_signal(sample, true_sol, t):
    """Generates and preps our signal"""

    time_flag = False
    n, k = get_model_constants().get("n"), get_model_constants().get("k")
    if n is None:
        n = len(sample)

    sample = sample[:n].reshape(-1, 1)
    true_sol = true_sol[:n].reshape(-1, 1)

    D = Difference_Matrix(n, k)

    if t is not None:
        t = t[:n]
        D = Time_Difference_Matrix(D, t)
        time_flag = True

    return sample, true_sol, D, time_flag


def perform_cv(sample, D, lambda_p, t):
    """Perform Cross-Validation on Lambda Penalty"""

    cv_folds = get_simulation_constants().get("cv_folds")
    cv_iterations = get_simulation_constants().get("cv_iterations")

    # perform CV
    best_lambda = cross_validation(
        sample, D, lambda_p=lambda_p, t=t, cv_folds=cv_folds, cv_iterations=cv_iterations, verbose=False
    )

    if best_lambda is None:
        print("No Optimal lambda found via Cross Validation")

        if lambda_p is None:
            print("No predefined lambda_p provided")
            return
        else:
            print("Using predefined lambda_p")

    else:
        if lambda_p is None:
            lambda_p = best_lambda
        else:
            lambda_p = lambda_p * best_lambda

    return (lambda_p, best_lambda)


def write_to_files(sample, true_sol, sol, true_knots, knots, plot, lambda_p):
    """Write artifacts to mlflow"""
    # plot to visualize estimation
    if plot:
        plt.figure(figsize=(14, 12))
        plt.plot(true_sol, color="black", label="True Signal", lw=10)
        plt.plot(sample, color="blue", label="Noisy Sample", lw=0.5)
        plt.plot(sol, color="red", label="Reconstructed Estimate", lw=5)
        plt.legend()
        plt.title("Linear Trend Filtering Estimate on Noisy Sample")
        plt.savefig("data/images/tf.png")
        plt.close()

        plt.figure(figsize=(14, 12))
        plt.plot(true_sol, color="black", label="True Signal", lw=10)
        plt.plot(sol, color="red", label="Reconstructed Estimate", lw=5)
        # vertical lines for regime changes
        if knots:
            for knot in knots:
                plt.axvline(x=knot, color="purple", linestyle="--", lw=2.5, label="Estimated Regime Change")

            for knot in true_knots:
                plt.axvline(x=knot, color="black", linestyle="--", lw=2.5, label="True Regime Change")

        plt.title("Estimated Regime Changes")
        plt.xlabel("Time")
        plt.ylabel("Observation")
        plt.savefig("data/images/knots.png")
        plt.close()

    # save files (eventually refactor custom model)
    with open("data/true_sol.txt", "w") as f:
        f.write(str(true_sol))

    with open("data/noisy_sample.txt", "w") as f:
        f.write(str(sample))

    with open("data/sol.txt", "w") as f:
        f.write(str(sol))

    if knots:
        with open("data/knots.txt", "w") as f:
            f.write(str(knots))

        with open("data/true_knots.txt", "w") as f:
            f.write(str(true_knots))

    if isinstance(lambda_p, np.ndarray):
        with open("data/lambda_p.txt", "w") as f:
            f.write(str(lambda_p))


def log_to_mlflow(
    exp_name,
    results,
    lambda_p,
    best_lambda,
    mse_from_sample,
    mse_from_true,
    expected_prediction_error,
    hausdorff_distance,
    len_true_knots,
    len_reconstructed_knots,
    flags,
):
    """Logs params, metrics, and tags to mlflow"""

    log_mlflow, bulk, include_cv = map(flags.get, ["log_mlflow", "bulk", "include_cv"])

    adaptive_penalty = isinstance(lambda_p, np.ndarray)

    # extract params and constants for logging
    cv_folds, cross_validation_size, reference_variance, signal_to_noise = map(
        get_simulation_constants().get, ["cv_folds", "cross_validation_size", "reference_variance", "signal_to_noise"]
    )

    k, n, maxiter, maxlsiter, tol, K_max, order = map(
        get_model_constants().get, ["k", "n", "maxiter", "maxlsiter", "tol", "K_max", "order"]
    )

    description = (
        "Linear Trend Filtering on Noisy Sample with Cross Validation of {cv_folds} folds and "
        "Cross Validation Size of {cross_validation_size}  Reference Variance of {reference_variance} "
        " Signal to Noise Ratio of {signal_to_noise} and Adaptive Penalty of {adaptive_penalty}".format(
            cv_folds=cv_folds,
            cross_validation_size=cross_validation_size,
            reference_variance=reference_variance,
            signal_to_noise=signal_to_noise,
            adaptive_penalty=adaptive_penalty,
        )
    )

    # create mlflow experiement (if not exists) and run
    experiment_id, run, run_tag = create_mlflow_experiment(exp_name, description=description, bulk=bulk)
    if log_mlflow:
        # Log params, metrics, tags, artifacts
        run_end = log_mlflow_params(
            run,
            params={
                "n": n,
                "k": k,
                "order": order,
                "maxiter": maxiter,
                "maxsliter": maxlsiter,
                "tol": tol,
                "cross_validation": include_cv,
                "no_folds": cv_folds,
                "cross_validation_size": cross_validation_size,
                "adaptive_lambda_p": adaptive_penalty,
                "signal_to_noise": signal_to_noise,
                "reference_variance": reference_variance,
                "k_max": K_max,
            },
            metrics={
                "computation_time": results["computation_time"],
                "optimal_relative_lambda": best_lambda,
                "mse_from_sample": mse_from_sample,
                "mse_from_true": mse_from_true,
                "hausdorff_distance": hausdorff_distance,
                "integrated_squared_prediction_error": expected_prediction_error,
                "len_true_knots": len_true_knots,
                "len_reconstructed_knots": len_reconstructed_knots,
                "knot_difference": len_true_knots - len_reconstructed_knots,
                "gap": results["gap"],
            },
            tags=[{"Adaptive": adaptive_penalty}, {"Cross_Validation": include_cv}, {"Status": results["status"]}],
            artifact_list=[
                "data/images/tf.png",
                "data/images/knots.png",
                "data/true_sol.txt",
                "data/noisy_sample.txt",
                "data/sol.txt",
                "data/knots.txt",
                "data/true_knots.txt",
                "data/lambda_p.txt",
            ],
        )
