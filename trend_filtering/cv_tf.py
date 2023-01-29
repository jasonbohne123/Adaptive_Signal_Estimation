from collections import defaultdict
from typing import Union

import numpy as np

from evaluation_metrics.loss_functions import compute_error
from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.helpers import compute_lambda_max
from trend_filtering.tf_constants import get_simulation_constants

### TO-DO Create Cross Validation Class


def cross_validation(
    x: np.ndarray,
    D: Difference_Matrix,
    lambda_p: Union[None, np.ndarray] = None,
    t: Union[None, np.ndarray] = None,
    cv_folds: int = None,
    cv_iterations: int = None,
    verbose=True,
):
    """Cross Validation for constant TF penalty parameter lambda_p"""

    n = len(x)
    cv_size = int(n * get_simulation_constants()["cross_validation_size"])

    # relative exponential grid
    grid = np.geomspace(get_simulation_constants()["cv_grid_lb"], 1, cv_folds)

    # initialize dictionary to store results
    results = defaultdict(float)

    # iterate over multiple cross validation indices to prevent overfitting  to oos data
    for i in range(cv_iterations):
        if verbose:
            print(f"Performing  {i} out of {cv_iterations} iterations of cross validation")

        # get in-sample and out-of-sample indices per each iteration
        # (randomly select cv_size indices which is ideal otherwise is just extrapolation)
        is_index = np.sort(np.random.choice(n, size=cv_size, replace=False))
        oos_index = np.sort(np.setdiff1d(np.arange(n), is_index))

        # get in-sample and out-of-sample data
        x_is = x[is_index]
        x_oos = x[oos_index]

        m = len(is_index)

        # compute difference matrix (more stable if equal time)

        D = Difference_Matrix(m, D.k)

        # compute lambda_max for each subproblem
        lambda_max = compute_lambda_max(D, x_is, time=False)

        for lambda_i in grid:

            lambda_scaler = lambda_i * lambda_max

            if verbose:
                print(f"Performing cross validation for lambda = {lambda_scaler}")

            # if prior is provided, scale lambda to have mean of candidate lambda
            if lambda_p is not None:

                # must be multivariate ndarray if not None
                assert isinstance(lambda_p, np.ndarray)

                # scale penalty to have mean of optimal lambda
                # (is mean the best statistic here)

                padded_lambda_p = np.pad(lambda_p, (1, 1), "mean")

                lambda_p_is = padded_lambda_p[is_index][1:-1]

                lambda_scaler = lambda_p_is * lambda_scaler / np.mean(lambda_p_is)

            # solve tf subproblem
            result = adaptive_tf(x_is.reshape(-1, 1), D, t=None, lambda_p=lambda_scaler)
            status = result["status"]
            sol = result["sol"]

            if sol is None:
                if verbose:
                    print("No solution found for lambda = {}".format(lambda_scaler))
                    print("Status: {}".format(status))

                # ignore cases where no solution is found
                results[lambda_i] += np.inf
                continue

            # to compute oos error we need to make the return type callable
            predictions = sol.predict(oos_index)

            # compute mse on oos test set
            oos_error = compute_error(predictions, x_oos, type="mse")

            # add to average oos error for each lambda
            results[lambda_i] += oos_error

    # get best lambda from all iterations
    best_lambda_dict = {k: v / cv_iterations for k, v in results.items()}
    best_lambda = min(best_lambda_dict, key=best_lambda_dict.get)

    # compute lambda_max for original problem
    D = Difference_Matrix(n, D.k)

    if t is None:
        orig_lambda_max = compute_lambda_max(D, x, time=False)
    else:
        T = Time_Difference_Matrix(D, t=t)
        orig_lambda_max = compute_lambda_max(T, x, time=True)

    return best_lambda * orig_lambda_max
