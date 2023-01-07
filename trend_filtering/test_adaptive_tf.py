import matplotlib.pyplot as plt
import numpy as np

from trend_filtering.adaptive_tf import adaptive_tf


def test_adaptive_tf(x, n=None, plot=False):
    """Test adaptive_tf function"""

    # generate signal
    if n is None:
        n = len(x)

    x = x[:n]

    lambda_p = np.array([1.0])
    # reconstruct signal
    x_hat, status, gap = adaptive_tf(x.reshape(-1, 1), lambda_p=lambda_p, verbose=False)

    if plot:
        # plot
        plt.figure(figsize=(10, 4))
        plt.plot(x, "b", label="noisy signal")
        plt.plot(x_hat, "r", label="reconstructed signal")
        plt.legend()
        plt.title("Reconstruction of a noisy signal with adaptive TF penalty")
        plt.show()

    return
