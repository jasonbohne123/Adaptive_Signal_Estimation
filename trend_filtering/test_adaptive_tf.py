import time
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from simulations.mlflow_helpers import create_mlflow_experiment, log_mlflow_params
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.cv_tf import cross_validation
from trend_filtering.helpers import compute_error, compute_lambda_max
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

    cv_folds, sample_variance = map(get_simulation_constants().get, ["cv_folds", "sample_variance"])
    k, n, maxiter, maxlsiter, tol = map(get_model_constants().get, ["k", "n", "maxiter", "maxlsiter", "tol"])
    include_cv, plot, verbose, bulk, log_mlflow = map(
        flags.get, ["include_cv", "plot", "verbose", "bulk", "log_mlflow"]
    )

    time_flag = False

    # generate signal
    if n is None:
        n = len(sample)

    sample = sample[:n].reshape(-1, 1)
    true_sol = true_sol[:n].reshape(-1, 1)

    D = Difference_Matrix(n, k)

    if t is not None:
        t = t[:n]
        D = Time_Difference_Matrix(D, t)
        time_flag = True

    if not include_cv and not lambda_p:
        print(" No lambda_p provided and no cross validation")
        return

    # cross validation grid
    if include_cv:

        # compute lambda_max to know grid boundary
        lambda_max = compute_lambda_max(D, sample, time=time_flag)

        # exponential grid
        grid = np.geomspace(0.0001, lambda_max, cv_folds)

        # perform CV
        optimal_lambda, gap = cross_validation(sample, D, grid=grid, t=None, verbose=False)

        if optimal_lambda is None:
            print("No Optimal lambda found via Cross Validation")

            if lambda_p is None:
                print("No predefined lambda_p provided")
                return
            else:
                print("Using predefined lambda_p")

        else:
            lambda_p = optimal_lambda

    # check if penalty is adaptive
    adaptive_penalty = True if isinstance(lambda_p, np.ndarray) else False

    # reconstruct signal
    results = adaptive_tf(sample, D_=D, t=t, lambda_p=lambda_p)

    if verbose:
        print(f"Solved TF problem with status: {results['status']}")
    results["computation_time"] = time.time() - start_time

    # extract solution information
    gap = results["gap"]
    status = results["status"]
    sol = results["sol"]

    if sol is not None:
        sol = sol.reshape(-1, 1)
    else:
        print("No solution found")
        return

    # compute mse from sample and true
    mse_from_sample = compute_error(sample, sol, type="mse")
    mse_from_true = compute_error(true_sol, sol, type="mse")

    # plot to visualize estimation
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(true_sol, color="orange", label="True Signal", lw=1.25)
        plt.plot(sample, color="blue", label="Noisy Sample", lw=2)
        plt.plot(sol, color="red", label="Reconstructed Estimate", lw=1.25)
        plt.legend()
        plt.title("Reconstruction of a noisy signal with TF penalty")
        plt.savefig("data/images/tf.png")
        plt.close()

    # save files (eventually refactor custom model)
    with open("data/true_sol.txt", "w") as f:
        f.write(str(true_sol))

    with open("data/noisy_sample.txt", "w") as f:
        f.write(str(sample))

    with open("data/sol.txt", "w") as f:
        f.write(str(sol))

    # create mlflow experiement (if not exists) and run
    experiment_id, run, run_tag = create_mlflow_experiment(exp_name, bulk=bulk)
    if log_mlflow:
        # Log params, metrics, tags, artifacts
        run_end = log_mlflow_params(
            run,
            params={
                "n": n,
                "k": k,
                "maxiter": maxiter,
                "maxsliter": maxlsiter,
                "tol": tol,
                "cross_validation": include_cv,
                "no_folds": cv_folds,
                "adaptive_lambda_p": adaptive_penalty,
                "sample_variance": sample_variance,
            },
            metrics={
                "computation_time": results["computation_time"],
                "lambda_max": lambda_max,
                "optimal_lambda": optimal_lambda,
                "mse_from_sample": mse_from_sample,
                "mse_from_true": mse_from_true,
                "gap": gap,
            },
            tags=[{"Adaptive": adaptive_penalty}, {"Cross_Validation": include_cv}, {"Status": status}],
            artifact_list=["data/images/tf.png", "data/true_sol.txt", "data/noisy_sample.txt", "data/sol.txt"],
        )

    return
