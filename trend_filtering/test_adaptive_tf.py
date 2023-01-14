import time
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from simulations.mlflow_helpers import create_mlflow_experiment, log_mlflow_params
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.cv_tf import cross_validation
from trend_filtering.helpers import compute_error
from trend_filtering.tf_constants import get_model_constants, get_simulation_constants


def test_adaptive_tf(
    sample: np.ndarray,
    t: Union[None, np.ndarray] = None,
    true_sol: Union[None, np.ndarray] = None,
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
        (
            lambda_p,
            best_lambda,
            lambda_max,
            best_oos_error,
            optimal_predictions,
            optimal_estimate,
            observed,
            oos_index,
        ) = perform_cv(sample, D, lambda_p, t)

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

    # write artifacts to files
    write_to_files(
        sample, true_sol, sol_array, knots, plot, lambda_p, optimal_predictions, optimal_estimate, observed, oos_index
    )

    # log information to mlflow
    if log_mlflow:
        log_to_mlflow(
            exp_name, results, lambda_p, lambda_max, best_lambda, best_oos_error, mse_from_sample, mse_from_true, flags
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
    # perform CV
    (
        best_lambda,
        lambda_max,
        best_oos_error,
        optimal_predictions,
        optimal_estimate,
        observed,
        oos_index,
    ) = cross_validation(sample, D, lambda_p=lambda_p, t=t, cv_folds=cv_folds, verbose=False)

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

    return lambda_p, best_lambda, lambda_max, best_oos_error, optimal_predictions, optimal_estimate, observed, oos_index


def write_to_files(sample, true_sol, sol, knots, plot, lambda_p, op, oe, obs, oos_index):
    """Write artifacts to mlflow"""
    # plot to visualize estimation
    if plot:
        plt.figure(figsize=(14, 12))
        plt.plot(true_sol, color="orange", label="True Signal", lw=4.0)
        plt.plot(sample, color="blue", label="Noisy Sample", lw=0.25)
        plt.plot(sol, color="red", label="Reconstructed Estimate", lw=1.25)
        plt.plot(oe, color="green", label="Optimal I.S. Estimate", lw=1.25)
        plt.scatter(oos_index, op, color="green", label="Optimal Prediction", lw=0.75)
        plt.scatter(oos_index, obs, color="black", label="Observed", lw=0.75)
        if knots:
            plt.scatter(knots, sol[knots], color="purple", label="Knots", marker="*", lw=10.0)
        plt.legend()
        plt.title("Linear Trend Filtering Estimate on Noisy Sample")
        plt.savefig("data/images/tf.png")
        plt.close()

    # save files (eventually refactor custom model)
    with open("data/true_sol.txt", "w") as f:
        f.write(str(true_sol))

    with open("data/noisy_sample.txt", "w") as f:
        f.write(str(sample))

    with open("data/sol.txt", "w") as f:
        f.write(str(sol))

    with open("data/optimal_predictions.txt", "w") as f:
        f.write(str(op))

    with open("data/observed.txt", "w") as f:
        f.write(str(obs))

    with open("data/optimal_estimate.txt", "w") as f:
        f.write(str(oe))

    if knots:
        with open("data/knots.txt", "w") as f:
            f.write(str(knots))

    if isinstance(lambda_p, np.ndarray):
        with open("data/lambda_p.txt", "w") as f:
            f.write(str(lambda_p))


def log_to_mlflow(
    exp_name, results, lambda_p, lambda_max, best_lambda, best_oos_error, mse_from_sample, mse_from_true, flags
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
                "lambda_max": lambda_max,
                "optimal_lambda": best_lambda,
                "oos_error": best_oos_error,
                "mse_from_sample": mse_from_sample,
                "mse_from_true": mse_from_true,
                "gap": results["gap"],
            },
            tags=[{"Adaptive": adaptive_penalty}, {"Cross_Validation": include_cv}, {"Status": results["status"]}],
            artifact_list=[
                "data/images/tf.png",
                "data/true_sol.txt",
                "data/noisy_sample.txt",
                "data/sol.txt",
                "data/optimal_predictions.txt",
                "data/observed.txt",
                "data/optimal_estimate.txt",
            ],
        )
