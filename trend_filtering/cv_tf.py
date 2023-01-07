import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.adaptive_tf import adaptive_tf


def cross_validation(y, D: Difference_Matrix, grid, verbose=True):
    """Cross Validation for constant TF penalty parameter lambda_p"""

    best_gap = np.inf
    best_lambda = None

    for lambda_i in grid:
        result = adaptive_tf(y.reshape(-1, 1), D, lambda_p=[lambda_i], verbose=False)

        # extract solution information
        gap = result["gap"]
        status = result["status"]
        sol = result["sol"]

        if sol is None:
            if verbose:
                print("No solution found for lambda = {}".format(lambda_i))
                print("Status: {}".format(status))
            continue

        if gap < best_gap:
            best_gap = gap
            best_lambda = lambda_i

    # if all cv failed
    if best_lambda is None:
        return None, None

    return best_lambda, best_gap
