import sys

sys.path.append("../")

import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.fused_lasso import fused_lasso


def specialized_admm(y: np.ndarray, D_: Difference_Matrix, lambda_: float):
    """Specialized ADMM Implementation for Trend Filtering"""

    # construct difference matrix of order k (not k-1) from original difference matrix
    n = y.shape[0]
    k = D_.k
    y = y.reshape(-1, 1)
    D_ = D_.compute_k_difference(k - 1)
    D = D_
    D_t_D = D.T.dot(D)

    # set max iterations
    MAX_ITER = 250

    # initialize variables with guesses
    beta = y.copy()

    # alpha is in R^{n-k-1 x 1}
    alpha = D.dot(beta)
    u = np.zeros([n - k, 1])

    # ref. sets rho to lambda for stability
    rho = lambda_

    # Pre-compute to save some multiplications
    I = np.identity(n)
    rho_D_T = rho * D.T
    Q = I + rho * D_t_D
    Q_inv = np.linalg.inv(Q)
    Q_inv_dot = Q_inv.dot

    # Calculate lambda_max (order k+1)
    D_k_1 = Difference_Matrix(n, k=k)
    np.amax(np.absolute(np.linalg.inv(D_k_1.D.dot(D_k_1.D.T)).dot((D_k_1.D).dot(y))))

    for _ in range(MAX_ITER):

        # beta is in R^{n x 1}
        beta = Q_inv_dot(y + rho_D_T.dot((alpha + u).reshape(-1, 1)))

        # precompute observation matrix
        D_k_beta_u = (D @ beta - u).flatten()

        # alpha is in R^{n-k-1 x 1}
        alpha = fused_lasso(D_k_beta_u, alpha, n, k - 1, lambda_, rho)

        # update u
        u = u + alpha - D.dot(beta)

    return beta
