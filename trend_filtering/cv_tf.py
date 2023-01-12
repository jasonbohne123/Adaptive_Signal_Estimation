from typing import Union

import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.adaptive_tf import adaptive_tf


def cross_validation(
    y: np.ndarray, D: Difference_Matrix, grid: np.ndarray, lambda_p:Union[None,np.ndarray]=None,t: Union[None, np.ndarray] = None, verbose=True
):
    """Cross Validation for constant TF penalty parameter lambda_p"""

    best_gap = np.inf
    best_lambda = None
    for lambda_i in grid:
        
        if lambda_p is not None:
            
            # must be multivariate ndarray if not None
            assert isinstance(lambda_p, np.ndarray)

            lambda_i = lambda_p*lambda_i

        result = adaptive_tf(y.reshape(-1, 1), D, t, lambda_p=lambda_i)

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
