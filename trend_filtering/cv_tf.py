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
    prior: Union[None, np.ndarray] = None,
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
        prior_max = compute_lambda_max(D, x_is, time=False)

        for lambda_i in grid:

            best_scaler = lambda_i * prior_max

            if verbose:
                print(f"Performing cross validation for lambda = {best_scaler}")

            # if prior is provided, scale lambda to have mean of candidate lambda
            if prior is not None:

                # must be multivariate ndarray if not None
                assert isinstance(prior, np.ndarray)

                # scale penalty to have mean of optimal lambda
                # (is mean the best statistic here)

                padded_prior = np.pad(prior, (1, 1), "mean")

                prior_is = padded_prior[is_index][1:-1]

                best_scaler = prior_is * best_scaler / np.mean(prior_is)

            # solve tf subproblem
            result = adaptive_tf(x_is.reshape(-1, 1), D, t=None, prior=best_scaler)
            status = result["status"]
            sol = result["sol"]

            if sol is None:
                if verbose:
                    print("No solution found for lambda = {}".format(best_scaler))
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
    best_prior_dict = {k: v / cv_iterations for k, v in results.items()}
    best_prior = min(best_prior_dict, key=best_prior_dict.get)

    # compute lambda_max for original problem
    D = Difference_Matrix(n, D.k)

    if t is None:
        orig_scaler_max = compute_lambda_max(D, x, time=False)
    else:
        T = Time_Difference_Matrix(D, t=t)
        orig_scaler_max = compute_lambda_max(T, x, time=True)

    return best_prior * orig_scaler_max
