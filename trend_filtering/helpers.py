import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.tf_constants import get_model_constants


def compute_lambda_max(D: Difference_Matrix, x: np.ndarray):
    """Computes the maximum lambda value for the adaptive trend filtering algorithm"""

    DDT_inv = np.linalg.solve(D.DDT, np.eye(D.DDT.shape[0]))

    max_error = np.max(abs(D.DDT.dot(DDT_inv) - np.eye(D.DDT.shape[0])))

    if max_error > 1e-6:
        print(f"Warning: DDT is not invertible. Max error: {max_error}")

    D_D = D.D

    # lambda value which gives best affine fit
    lambda_max = np.max(abs(DDT_inv.dot(D_D).dot(x)))

    return lambda_max, D


def extract_cp(smooth, D: Difference_Matrix, quantile):
    """Extract changepoints via difference operator"""

    n = D.n
    D = D.D

    diff = np.dot(D, smooth).reshape(1, -1)[0]

    # get cp
    points = np.sort(np.abs(diff))

    # get quantile
    threshold = points[int((1 - quantile) * len(points))]

    # get index of cp
    index = np.where(np.abs(diff) > threshold)[0]

    # returns rhs of index for close pairs
    min_cp_distance = int(get_model_constants()["min_cp_distance"] * n)
    close_pairs = index[1:][np.diff(index) < min_cp_distance]

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
