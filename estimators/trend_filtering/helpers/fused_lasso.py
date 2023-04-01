import cvxpy as cvx
import numpy as np


def fused_lasso(D_k_beta_u, alpha, n, k, lambda_, rho):
    """Fused Lasso Wrapper Adapted to ADMM"""

    # feature matrix is identity matrix
    X = np.eye(n - k - 1)

    # cvxpy variables
    alpha_hat = cvx.Variable(n - k - 1)

    # warm start
    alpha_hat.value = alpha.flatten()

    # generate difference matrix
    if k < 0:
        forwardDiff = alpha_hat[1:] - alpha_hat[:-1]
    else:
        # second difference is difference of first difference
        for i in range(0, k + 1):
            forwardDiff = alpha_hat[1:] - alpha_hat[:-1]

    # fused lasso objective of mse + lambda/rho*norm
    objective = cvx.Minimize(
        0.5 * cvx.sum_squares(X @ alpha_hat - D_k_beta_u) + lambda_ / rho * cvx.norm(forwardDiff, 1)
    )

    prob = cvx.Problem(objective)

    result = prob.solve(solver=cvx.OSQP)

    return alpha_hat.value.reshape(-1, 1)


### Ref. Available Solvers within CVXPY
# https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
