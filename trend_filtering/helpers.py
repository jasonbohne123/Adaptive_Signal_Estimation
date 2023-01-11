import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix


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


def extract_cp(smooth, k=2, threshold=1e-6):
    """Extract changepoints via difference operator"""
    n = len(smooth)
    diff_mat = Difference_Matrix(n, k)
    D = diff_mat.D
    diff = np.dot(D, smooth).reshape(1, -1)[0]

    x, y, index = np.where([abs(diff) > threshold])
    return index


def compute_error(x, x_hat, type="mse"):
    assert type in ["mse", "mae"]

    assert x.shape == x_hat.shape

    if type == "mse":
        return np.sum(np.abs(x - x_hat) ** 2) / len(x)

    elif type == "mae":
        return np.sum(np.abs(x - x_hat)) / len(x)

    else:
        raise ValueError("Error type not recognized")
