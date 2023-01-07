import matplotlib.pyplot as plt
import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.cv_tf import cv_tf_penalty
from trend_filtering.helpers import compute_lambda_max


def test_adaptive_tf(x=None, n=None, plot=False, include_cv=False, lambda_p=None):
    """Test adaptive_tf function"""

    if x is None:
        x = np.sin(np.linspace(0, 10, 500)) + np.random.randn(500) * 0.1
        n = 500

    # generate signal
    if n is None:
        n = len(x)

    x = x[:n]

    D = Difference_Matrix(n, 2)

    # cross validation between 0 and lambda_max (if specified)
    if include_cv:
        compute_lambda_max(D)
        grid = [0.5]  # np linspace(0, lambda_max, 10)
        lambda_p, gap = cv_tf_penalty(x, D, grid, verbose=True)

    if lambda_p is None:
        lambda_p = np.array([0.5])

    # reconstruct signal
    result = adaptive_tf(x, D, lambda_p=lambda_p, verbose=True)

    if result is None:
        return
    sol = result["sol"]
    result["gap"]

    if plot:
        # plot
        plt.figure(figsize=(10, 4))
        plt.plot(x, "b", label="noisy signal")
        plt.plot(sol, "r", label="reconstructed signal")
        plt.legend()
        plt.title("Reconstruction of a noisy signal with adaptive TF penalty")
        plt.show()

    return
