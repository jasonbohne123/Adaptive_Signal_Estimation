import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.tf_constants import get_model_constants


def compute_lambda_max(D: Difference_Matrix, x: np.ndarray):
    """Computes the maximum lambda value for the adaptive trend filtering algorithm"""

    DDT_inv = np.linalg.solve(D.DDT, np.eye(D.DDT.shape[0]))
    D_D = D.D

    # lambda value which gives best affine fit
    lambda_max = np.max(abs(DDT_inv.dot(D_D).dot(x)))

    return lambda_max, D


def extract_cp(smooth, D: Difference_Matrix, threshold):
    """Extract changepoints via difference operator"""

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
