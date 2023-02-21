from collections import defaultdict

import numpy as np

from evaluation_metrics.loss_functions import compute_error
from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.helpers import compute_lambda_max
from trend_filtering.tf_constants import get_simulation_constants


def cross_validation(
    x: np.ndarray,
    D: Difference_Matrix,
    cv_folds: int = None,
    cv_iterations: int = None,
    verbose=True,
):
    """Cross Validation for constant TF penalty parameter lambda_p"""

    cv_size = int(len(x) * get_simulation_constants()["cross_validation_size"])

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
        is_index = np.sort(np.random.choice(len(x), size=cv_size, replace=False))
        oos_index = np.sort(np.setdiff1d(np.arange(len(x)), is_index))

        # get in-sample and out-of-sample data
        x_is = x[is_index]
        x_oos = x[oos_index]

        is_t = D.t[is_index] if D.time_enabled else None
        is_prior = D.prior[is_index] if D.prior_enabled else None

        is_D = Difference_Matrix(len(is_index), D.k, is_t, is_prior)  # normalize prior by mean

        # compute lambda_max for each subproblem given difference matrix and prior
        prior_max, is_D_D = compute_lambda_max(is_D, x_is)

        for lambda_i in grid:

            # relative lambda scaled to max
            relative_scaler = lambda_i * prior_max

            if verbose:
                print(f"Performing cross validation for lambda = {relative_scaler}")

            # solve tf subproblem
            result = adaptive_tf(x_is.reshape(-1, 1), is_D_D, lambda_p=relative_scaler)
            status = result["status"]
            sol = result["sol"]

            if sol is None:
                if verbose:
                    print("No solution found for lambda = {}".format(relative_scaler))
                    print("Status: {}".format(status))

                # ignore cases where no solution is found
                results[lambda_i] += np.inf
                continue

            # to compute oos error we need to make the return type callable
            predictions = sol.predict(oos_index)

            # compute mse on oos test set

            oos_error = compute_error(predictions.reshape(-1, 1), x_oos, type="mse")

            # add to average oos error for each lambda
            results[lambda_i] += oos_error

    # get best lambda from all iterations
    best_prior_dict = {k: v / cv_iterations for k, v in results.items()}
    best_prior = min(best_prior_dict, key=best_prior_dict.get)

    # get original lambda_max
    is_t = D.t if D.time_enabled else None
    orig_scaler_max, D_bar = compute_lambda_max(D, x)

    return best_prior * orig_scaler_max


def perform_cv(
    sample,
    D: Difference_Matrix,
):
    """Perform Cross-Validation on Lambda Penalty"""

    cv_folds = get_simulation_constants().get("cv_folds")
    cv_iterations = get_simulation_constants().get("cv_iterations")
    verbose_cv = get_simulation_constants().get("verbose_cv")

    # best relative lambda scaled by lambda_max
    scaled_prior = cross_validation(sample, D, cv_folds=cv_folds, cv_iterations=cv_iterations, verbose=verbose_cv)

    return scaled_prior
