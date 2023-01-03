import numpy as np

from trend_filtering.adaptive_tf import adaptive_tf


def cv_tf_penalty(y, grid, t=None, verbose=True):
    """Cross Validation for constant TF penalty parameter lambda_p"""

    best_gap = np.inf
    best_lambda = None

    # construct D matrix and inverse ; pass into the algorithm as a parameter

    for lambda_i in grid:
        x, status, gap = adaptive_tf(y, t, [lambda_i], verbose=verbose)

        if gap < best_gap:
            best_gap = gap
            best_lambda = lambda_i

    return np.array([best_lambda]), np.array([best_gap])
