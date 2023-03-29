import sys

sys.path.append("../")
import cvxopt as cvx
import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix


def specialized_admm(y: np.ndarray, k: int = 1, lambda_: float = 1):
    """Specialized ADMM Implementation for Trend Filtering

    Ref: https://arxiv.org/abs/1406.2082
    """

    n = y.shape[0]  # n-k-1 total differences
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

    ## Fused Lasso

    def diff_op(shape):
        mat = np.zeros(shape)
        mat[range(2, shape[0]), range(1, shape[1])] = 1
        mat -= np.eye(shape)
        return mat

    def difference_pen(beta, epsilon):
        return cvx.norm1(diff_op(beta.shape) @ beta)

    for _ in range(MAX_ITER):
        # x minimisation step via posterier OLS
        beta = Q_inv_dot(y + rho_D_t.dot(alpha - u))

        alpha = difference_pen(beta, lambda_)

        # mulitplier update
        u = u + alpha - D_.dot(beta)

    return alpha
