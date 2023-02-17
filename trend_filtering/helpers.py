from typing import Union

import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from trend_filtering.tf_constants import get_model_constants


def compute_lambda_max(D: Union[Difference_Matrix, Time_Difference_Matrix], x: np.ndarray, t=None):
    """Computes the maximum lambda value for the adaptive trend filtering algorithm"""

    assert D.time_enabled == (t is not None), "Time must be provided if time enabled"

    if D.time_enabled:
        D = Difference_Matrix(len(x), k=D.k)
        D = Time_Difference_Matrix(D=D, t=t)

        DDT_inv = D.T_DDT_inv
        D_D = D.T_D

    else:
        D = Difference_Matrix(len(x), k=D.k)
        DDT_inv = D.DDT_inv
        D_D = D.D

    # lambda value which gives best affine fit
    lambda_max = np.max(abs(DDT_inv.dot(D_D).dot(x)))

    return lambda_max, D


def extract_cp(smooth, D: Union[Difference_Matrix, Time_Difference_Matrix], threshold):
    """Extract changepoints via difference operator"""

    # if time enabled, use time difference matrix
    if D.time_enabled:
        D = D.T_D
    else:
        D = D.D

    diff = np.dot(D, smooth).reshape(1, -1)[0]

    x, index = np.where([abs(diff) > threshold])

    # returns rhs of index for close pairs
    close_pairs = index[1:][np.diff(index) < get_model_constants()["min_cp_distance"]]

    index_to_remove = []

    # remove close pairs
    for rhs_index in close_pairs:
        i = diff[index[index < rhs_index][-1]]
        j = diff[rhs_index]

        # remove index with smaller absolute value
        if abs(i) > abs(j):
            index_to_remove.append(index[index < rhs_index][-1])

        # remove index with smaller absolute value
        else:
            index_to_remove.append(rhs_index)

    # return index with close pairs removed
    index = np.setdiff1d(index, index_to_remove)
    return index
