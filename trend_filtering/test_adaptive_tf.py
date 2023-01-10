import time
from typing import Dict, Union

import matplotlib.pyplot as plt
import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from simulations.mlflow_helpers import create_mlflow_experiment, log_mlflow_params
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.cv_tf import cross_validation
from trend_filtering.helpers import compute_lambda_max


def test_adaptive_tf(
    x: np.ndarray,
    t: Union[None, np.ndarray] = None,
    lambda_p=Union[float, np.ndarray],
    n: int = None,
    k: int = 2,
    p: int = 10,
    exp_name="DEFAULT",
    flags: Dict[str, bool] = None,
):
    """Test adaptive_tf function"""

    include_cv, plot, verbose, bulk, log_mlflow = map(
        flags.get, ["include_cv", "plot", "verbose", "bulk", "log_mlflow"]
    )

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
        lambda_max = compute_lambda_max(D, x)
        grid = np.linspace(0.0001, lambda_max, p)
        optimal_lambda, gap = cross_validation(x, D, grid=grid, t=None, verbose=False)

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
    if verbose:
        print(f"Solved TF problem with status: {results['status']}")
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
        plt.savefig("images/adaptive_tf.png")
        plt.close()

    experiment_id, run, run_tag = create_mlflow_experiment(exp_name, bulk=bulk)
    if log_mlflow:
        # Log to MLFlow
        run_end = log_mlflow_params(
            run,
            {
                "n": n,
                "k": k,
                "gap": results["gap"],
                "mse": None,
                "cross_validation": include_cv,
                "no_folds": p,
                "adaptive_lambda_p": adaptive_penalty,
                "computation_time": results["computation_time"],
            },
            artifact_list=["images/adaptive_tf.png"],
        )

    return
