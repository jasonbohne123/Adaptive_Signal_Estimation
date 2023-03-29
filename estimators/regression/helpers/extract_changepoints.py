import sys

sys.path.append("../../matrix_algorithms")
sys.path.append("../../estimators")

import numpy as np
from base_estimator import Base_Estimator
from difference_matrix import Difference_Matrix
from segmentation_constants import get_segmentation_constants


def extract_cp(estimator: Base_Estimator, D: Difference_Matrix, quantile):
    """Extract changepoints across an estimator via difference operator"""

    y_hat = estimator.y_hat
    n = D.n
    D = D.D

    diff = np.dot(D, y_hat).reshape(1, -1)[0]

    # get cp
    points = np.sort(np.abs(diff))

    # get quantile
    threshold = points[int((1 - quantile) * len(points))]

    # get index of cp
    index = np.where(np.abs(diff) > threshold)[0]

    # returns rhs of index for close pairs
    min_cp_distance = int(get_segmentation_constants()["min_cp_distance"] * n)
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
