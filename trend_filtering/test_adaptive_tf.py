import time
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from simulations.mlflow.mlflow_helpers import create_mlflow_experiment, log_mlflow_params
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.cv_tf import cross_validation
from trend_filtering.helpers import compute_lambda_max


def test_adaptive_tf(
    x: np.ndarray,
    t: Union[None, np.ndarray] = None,
    lambda_p=Union[float, np.ndarray],
    n: int = None,
    k: int = 2,
    p: int = 5,
    include_cv=False,
    plot=False,
    verbose=False,
    log_mlflow=False,
):
    """Test adaptive_tf function"""
    # generate signal
    if n is None:
        n = len(x)

    D = Difference_Matrix(n, k)

    x = x[:n]

    if t is not None:
        t = t[:n]

    adaptive_penalty = True if isinstance(lambda_p, np.ndarray) else False

    if not include_cv and not lambda_p:
        print(" No lambda_p provided and no cross validation")
        return

    if include_cv:
        # cross validation
        lambda_max = compute_lambda_max(D)
        grid = np.linspace(0.1, lambda_max, p)
        optimal_lambda, gap = cross_validation(x, D, grid=grid, t=None, verbose=verbose)

        if optimal_lambda is None:
            print("No Optimal lambda found via Cross Validation")

            if lambda_p is None:
                print("No predefined lambda_p provided")
                return
            else:
                print("Using predefined lambda_p")

        else:
            lambda_p = optimal_lambda

    # reconstruct signal
    start_time = time.time()
    results = adaptive_tf(x.reshape(-1, 1), D_=D, t=t, lambda_p=lambda_p)
    results["computation_time"] = time.time() - start_time

    # extract solution information
    results["gap"]
    results["status"]
    sol = results["sol"]

    if plot:
        # plot
        plt.figure(figsize=(10, 4))
        plt.plot(x, "b", label="noisy signal")
        plt.plot(sol, "r", label="reconstructed signal")
        plt.legend()
        plt.title("Reconstruction of a noisy signal with adaptive TF penalty")
        plt.show()
        plt.savefig("../simulations/images/adaptive_tf.png")

    run_id, tag = create_mlflow_experiment("L1TrendFiltering")
    if log_mlflow:
        # Log to MLFlow
        run = log_mlflow_params(
            run_id,
            {
                "n": n,
                "k": k,
                "gap": results["gap"],
                "cross_validation": include_cv,
                "no_folds": p,
                "adaptive_lambda_p": adaptive_penalty,
                "computation_time": results["computation_time"],
            },
            ["../simulations/images/adaptive_tf.png"],
        )

    return
