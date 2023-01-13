from typing import Union

import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.helpers import compute_error
from trend_filtering.tf_constants import get_simulation_constants


def cross_validation(
    x: np.ndarray,
    D: Difference_Matrix,
    grid: np.ndarray,
    lambda_p: Union[None, np.ndarray] = None,
    t: Union[None, np.ndarray] = None,
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

    # account for now unequally sized arrays
    D = Difference_Matrix(m, D.k)
    T = Time_Difference_Matrix(D, t=is_index)

    best_oos_error = np.inf
    best_lambda = None
    for lambda_i in grid:

        if lambda_p is not None:

            # must be multivariate ndarray if not None
            assert isinstance(lambda_p, np.ndarray)

            # scale penalty to have mean of optimal lambda
            # is mean the best statistic here?

            # Need to look into indexing here
            is_index_penalty = is_index[1:-1]
            lambda_i = lambda_p[is_index_penalty] * lambda_i / np.mean(lambda_p[is_index_penalty])

        result = adaptive_tf(x_is.reshape(-1, 1), T, t=is_index, lambda_p=lambda_i)
        status = result["status"]
        sol = result["sol"]

        if sol is None:
            if verbose:
                print("No solution found for lambda = {}".format(lambda_i))
                print("Status: {}".format(status))
            continue

        # compute oos error (Very Well might need to compute changepoints explicitly here )

        oos_error = compute_error(sol.predict(oos_index), x_oos, type="mse")
        print(oos_error)
        if best_oos_error > oos_error:
            best_oos_error = oos_error
            best_lambda = lambda_i

    # if all cv failed
    if best_lambda is None:
        return None, None

    return best_lambda, best_oos_error
