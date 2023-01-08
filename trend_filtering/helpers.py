import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix


def compute_lambda_max(diff_mat):
    """Computes the maximum lambda value for the adaptive trend filtering algorithm"""
    DDT_inv = diff_mat.DDT_inv
    lambda_max = np.sqrt(DDT_inv.diagonal().max())

    return lambda_max


def extract_cp(smooth, k=2, threshold=1e-6):
    """Extract changepoints via difference operator"""
    n = len(smooth)
    diff_mat = Difference_Matrix(n, k)
    D = diff_mat.D
    diff = np.dot(D, smooth).reshape(1, -1)[0]

    x, y, index = np.where([abs(diff) > threshold])
    return index
