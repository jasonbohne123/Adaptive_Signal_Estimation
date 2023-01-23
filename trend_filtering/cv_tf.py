from typing import Union

import numpy as np

from evaluation_metrics.loss_functions import compute_error
from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.helpers import compute_lambda_max
from trend_filtering.tf_constants import get_simulation_constants

### Create Cross Validation Class


def cross_validation(
    x: np.ndarray,
    D: Difference_Matrix,
    lambda_p: Union[None, np.ndarray] = None,
    t: Union[None, np.ndarray] = None,
    cv_folds: int = None,
    verbose=True,
):
    """Cross Validation for constant TF penalty parameter lambda_p"""

    n = len(x)

    # get in-sample and out-of-sample indices
    is_index = np.sort(
        np.random.choice(n, size=int(n * get_simulation_constants()["cross_validation_size"]), replace=False)
    )
    oos_index = np.sort(np.setdiff1d(np.arange(n), is_index))

    # get in-sample and out-of-sample data
    x_is = x[is_index]
    x_oos = x[oos_index]

    m = len(is_index)

    # account for now irregular time series
    if t is None:
        t = np.arange(n)
    D = Difference_Matrix(m, D.k)
    T = Time_Difference_Matrix(D, t=t[is_index])

    # compute lambda_max for exponential grid
    lambda_max = compute_lambda_max(T, x_is, time=True)

    # exponential grid
    grid = np.geomspace(get_simulation_constants()["cv_grid_lb"], lambda_max, cv_folds)

    best_oos_error = np.inf
    best_lambda, best_predictions, best_lambda = None, None, None
    observed = None

    for lambda_i in grid:

        lambda_scaler = lambda_i

        # if prior is provided, scale lambda to have mean of candidate lambda
        if lambda_p is not None:

            # must be multivariate ndarray if not None
            assert isinstance(lambda_p, np.ndarray)

            # scale penalty to have mean of optimal lambda
            # (is mean the best statistic here)

            padded_lambda_p = np.pad(lambda_p, (1, 1), "mean")

            lambda_p_is = padded_lambda_p[is_index][1:-1]

            lambda_i = lambda_p_is * lambda_i / np.mean(lambda_p_is)

        result = adaptive_tf(x_is.reshape(-1, 1), T, t=is_index, lambda_p=lambda_i)
        status = result["status"]
        sol = result["sol"]

        if sol is None:
            if verbose:
                print("No solution found for lambda = {}".format(lambda_i))
                print("Status: {}".format(status))
            continue

        # to compute oos error we need to make the return type callable
        predictions = sol.predict(oos_index)

        # compute mse on oos test set
        oos_error = compute_error(predictions, x_oos, type="mse")

        # if better than previous best, update
        if best_oos_error > oos_error:
            best_oos_error = oos_error
            best_lambda = lambda_scaler
            best_predictions = predictions
            observed = x_oos
            optimal_estimate = sol.x

    # if all cv failed
    if best_lambda is None:
        return None, None, None, None, None

    return best_lambda, lambda_max, best_oos_error, best_predictions, optimal_estimate, observed, is_index, oos_index
