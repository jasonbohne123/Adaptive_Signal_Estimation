from typing import Union

import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from trend_filtering.tf_constants import get_simulation_constants


def compute_lambda_max(diff_mat, y, time=False):
    """Computes the maximum lambda value for the adaptive trend filtering algorithm"""

    if time:
        DDT_inv = diff_mat.T_DDT_inv
        D = diff_mat.T_D
    else:
        DDT_inv = diff_mat.DDT_inv
        D = diff_mat.D

    # lambda value which gives best affine fit
    lambda_max = np.max(abs(DDT_inv.dot(D).dot(y)))

    return lambda_max


def extract_cp(smooth, D: Union[Difference_Matrix, Time_Difference_Matrix], threshold):
    """Extract changepoints via difference operator"""

    # if time enabled, use time difference matrix
    if D.time_enabled:
        D = D.T_D
    else:
        D = D.D

    diff = np.dot(D, smooth).reshape(1, -1)[0]

    x, index = np.where([abs(diff) > threshold])
    return index


def compute_error(x, x_hat, type="mse"):
    assert type in ["mse", "mae"]

    assert x.shape == x_hat.shape

    if type == "mse":
        return np.sum(np.abs(x - x_hat) ** 2) / len(x)

    elif type == "mae":
        return np.sum(np.abs(x - x_hat)) / len(x)

    elif type == "epe":
        var_y = get_simulation_constants().get("reference_variance")

        mse = compute_error(x, x_hat, type="mse")

        return var_y + mse

    else:
        raise ValueError("Error type not recognized")
