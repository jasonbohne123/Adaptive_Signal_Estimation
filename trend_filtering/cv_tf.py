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

    # dictionary to store average oos error for each relative lambda
    lambda_i_dict = defaultdict(float)

    # iterate over multiple cross validation indices to prevent overfitting  to oos data
    for i in range(cv_iterations):

        # get in-sample and out-of-sample indices per each iteration
        is_index = np.sort(np.random.choice(n, size=cv_size, replace=False))
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

        # compute lambda_max for exponential grid; specific to each index
        lambda_max = compute_lambda_max(T, x_is, time=True)

        for lambda_i in grid:

            # if prior is provided, scale lambda to have mean of candidate lambda
            if lambda_p is not None:

                # must be multivariate ndarray if not None
                assert isinstance(lambda_p, np.ndarray)

                # scale penalty to have mean of optimal lambda
                # (is mean the best statistic here)

                padded_lambda_p = np.pad(lambda_p, (1, 1), "mean")

                lambda_p_is = padded_lambda_p[is_index][1:-1]

                lambda_i = lambda_p_is * lambda_i / np.mean(lambda_p_is)

            # scale relative to lambda_max
            result = adaptive_tf(x_is.reshape(-1, 1), T, t=is_index, lambda_p=lambda_i * lambda_max)
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

            # add to average oos error for each lambda
            lambda_i_dict[lambda_i] += oos_error

    # get best lambda from average of all cv iterations
    best_lambda_dict = {k: v / cv_iterations for k, v in lambda_i_dict.items()}
    best_lambda = min(best_lambda_dict, key=best_lambda_dict.get)

    # rescale by lambda max of entire time series
    if t is None:
        t = np.arange(n)
    D = Difference_Matrix(n, D.k)
    T = Time_Difference_Matrix(D, t=t)

    lambda_max_all = compute_lambda_max(T, x, time=True)

    return best_lambda * lambda_max_all
