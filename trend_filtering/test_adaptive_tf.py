import matplotlib.pyplot as plt
import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.adaptive_tf import adaptive_tf


def test_adaptive_tf(x, n=None, k=2, plot=False):
    """Test adaptive_tf function"""

    # generate signal
    if n is None:
        n = len(x)

    D = Difference_Matrix(n, k)

    x = x[:n]

    lambda_p = np.array([1.0])
    # reconstruct signal
    x_hat, status, gap = adaptive_tf(x.reshape(-1, 1), D_=D, lambda_p=lambda_p, verbose=False)

    if plot:
        # plot
        plt.figure(figsize=(10, 4))
        plt.plot(x, "b", label="noisy signal")
        plt.plot(x_hat, "r", label="reconstructed signal")
        plt.legend()
        plt.title("Reconstruction of a noisy signal with adaptive TF penalty")
        plt.show()

    return
