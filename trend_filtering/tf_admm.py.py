import sys

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix


def specialized_admm(y: np.ndarray, k: int = 2):
    """Specialized ADMM Implementation for Trend Filtering

    Ref: https://arxiv.org/abs/1406.2082
    """

    n = y.shape[0]
    D = Difference_Matrix(n, k=k)
    D_ = D.D
    D_t_D = D_.T.dot(D_)

    w, v = np.linalg.eig(D_t_D)
    MAX_ITER = 10000

    # Function to caluculate min 1/2(y - Ax) + l||x||
    # via alternating direction methods
    beta = np.zeros([n, 1])
    alpha = np.zeros([n, 1])
    u = np.zeros([n, 1])

    # Calculate regression co-efficient and stepsize
    r = np.amax(np.absolute(w))
    rho = 1 / r

    # Pre-compute to save some multiplications
    I = np.identity(n)
    rho_D_t = rho * D_.T
    Q = I + rho * D_t_D
    Q_inv = np.linalg.inv(Q)

    Q_inv_dot = Q_inv.dot

    for _ in range(MAX_ITER):
        # x minimisation step via posterier OLS
        beta = Q_inv_dot(y + rho_D_t.dot(alpha - u))

        alpha = ...  # fused lasso

        # mulitplier update
        u = u + alpha - D_.dot(beta)

    return alpha


def test(n=200):
    """Test the ADMM method with randomly generated matrices and vectors"""

    x = np.linspace(0, 1, n)
    true_y = np.sin(2 * np.pi * x)

    # add noise
    y = true_y + np.random.normal(0, 0.1, n)

    # estimate trend
    trend = specialized_admm(y)

    # plot results
    plot(true_y, trend)

    plt.show()

    return


def plot(original, computed):
    """Plot two vectors to compare their values"""
    plt.plot(original, label="Original")
    plt.plot(computed, label="Estimate")

    plt.legend(loc="upper right")
