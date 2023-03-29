import numpy as np

from estimators.regression.helpers.segmentation_constants import get_segmentation_constants
from estimators.trend_filtering.helpers.difference_matrix import Difference_Matrix


def extract_cp(y: np.ndarray, D: Difference_Matrix, quantile):
    """Extract changepoints across an estimator via difference operator"""

    diff = np.dot(D.D, y).reshape(1, -1)[0]

    # get cp
    points = np.sort(np.abs(diff))

    # get quantile
    threshold = points[int((1 - quantile) * len(points))]

    # get index of cp
    index = np.where(np.abs(diff) > threshold)[0]

    # returns rhs of index for close pairs
    min_cp_distance = int(get_segmentation_constants()["min_cp_distance"] * D.n)
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
