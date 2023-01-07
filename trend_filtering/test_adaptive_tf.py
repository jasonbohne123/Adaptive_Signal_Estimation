import matplotlib.pyplot as plt
import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.adaptive_tf import adaptive_tf
from trend_filtering.cv_tf import cross_validation
from trend_filtering.helpers import compute_lambda_max


def test_adaptive_tf(x, n=None, k=2, include_cv=False, lambda_p=None, plot=False):
    """Test adaptive_tf function"""
    # generate signal
    if n is None:
        n = len(x)

    D = Difference_Matrix(n, k)

    x = x[:n]

    if not include_cv and not lambda_p:
        print(" No lambda_p provided and no cross validation")
        return

    if include_cv:
        # cross validation
        lambda_max = compute_lambda_max(D)
        grid = np.linspace(0.1, lambda_max, 10)
        optimal_lambda, gap = cross_validation(x, D, grid=grid, verbose=False)

        if optimal_lambda is None:
            print("No Optimal lambda found via Cross Validation")

            if lambda_p is None:
                print("No predefined lambda_p provided")
                return
            else:
                print("Using predefined lambda_p")
                lambda_p = np.array(lambda_p)

        lambda_p = optimal_lambda

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
