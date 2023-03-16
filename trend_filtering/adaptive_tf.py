from typing import Union

import numpy as np
from piecewise_linear_model import Piecewise_Linear_Model

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.tf_constants import get_model_constants

###### Numba Integration
# Utilize @njit decorator to compile functions to machine code
# Blocks default to object mode
# First compilation takes longer to cache the compilation of the code
# Utilies Intel's ICC compiler through Numba


def adaptive_tf(
    y: np.ndarray,
    D_: Difference_Matrix,
    lambda_p: Union[float, None] = 1.0,
    select_knots=False,
    true_knots=None,
    cv=False,
):
    """
    Adaptive trend filtering algorithm
    """

    hyperparams = get_model_constants()
    alpha, beta, mu, mu_inc, maxiter, maxlsiter, tol = map(
        hyperparams.get, ["alpha", "beta", "mu", "mu_inc", "maxiter", "maxlsiter", "tol"]
    )

    k = D_.k + 1
    n = len(y)

    m = n - k

    D, DDT = prep_difference_matrix(D_)

    Dy = np.dot(D, y)

    # init variables and objectives
    z = np.zeros((m, 1))
    mu1 = np.ones((m, 1))
    mu2 = np.ones((m, 1))

    step = np.inf

    lambda_p = lambda_p * np.ones((m, 1))

    f1 = z - lambda_p
    f2 = -z - lambda_p

    # main loop of iteration; solving a sequence of equality constrained quadratic programs
    for iters in range(maxiter + 1):

        DTz, DDTz, w = prep_matrices(D, Dy, z, mu1, mu2)

        # compute objectives
        pobj1, pobj2, dobj, gap = compute_objective(DDT, Dy, DTz, DDTz, z, w, mu1, mu2, lambda_p)

        # if duality gap becomes negative
        if gap < 0:
            status = "negative duality gap"
            x = y - np.dot(D.transpose(), z)
            return {"sol": None, "status": status, "gap": -1, "iters": iters}

        # if duality gap is small enough
        if gap <= tol:
            status = "solved"
            x = y - np.dot(D.transpose(), z)
            return {
                "sol": Piecewise_Linear_Model(x, D=D_, select_knots=select_knots, true_knots=true_knots),
                "status": status,
                "gap": gap,
                "iters": iters,
            }

        # update step
        newz, newmu1, newmu2, newf1, newf2 = update_step(
            DDT, DDTz, Dy, lambda_p, z, w, mu1, mu2, f1, f2, mu, mu_inc, step, gap, m, alpha, beta, maxlsiter
        )

        # adaptive stepsize of mu with ratio gamma
        # newmu1, newmu2 = adaptive_step_size(pobj1, pobj2, newmu1, newmu2, gamma)

        z = newz
        mu1 = newmu1
        mu2 = newmu2
        f1 = newf1
        f2 = newf2

    status = "maxiter exceeded"
    return {"sol": None, "status": status, "gap": -1, "iters": iters}


def prep_difference_matrix(D_: Difference_Matrix):
    """Accounts for irregular time series in difference matrix"""

    D = D_.D
    DDT = D_.DDT
    return D, DDT


def prep_matrices(D, Dy, z, mu1, mu2):
    """Prep matrices for objective computation"""

    DTz = np.dot(D.T, z)
    DDTz = np.dot(D, DTz)
    w = Dy - (mu1 - mu2)
    return DTz, DDTz, w


def compute_objective(DDT, Dy, DTz, DDTz, z, w, mu1, mu2, lambda_p):
    """Computes Primal and Dual objectives and duality gap"""

    # evaluates primal with dual variable of dual and optimality condition
    # linear_solver = np.linalg.solve(DDT, w)
    # np.max(np.abs(np.dot(DDT, linear_solver) - w))

    pobj1 = 0.5 * np.dot(w.T, (np.linalg.solve(DDT, w))) + np.sum(np.dot(lambda_p.T, (mu1 + mu2)))
    pobj2 = 0.5 * np.dot(DTz.transpose(), DTz) + np.sum(np.dot(lambda_p.T, np.abs(Dy - DDTz)))
    pobj1 = pobj1.item()
    pobj2 = pobj2.item()
    pobj = min(pobj1, pobj2)
    dobj = -0.5 * np.dot(DTz.transpose(), DTz) + np.dot(Dy.transpose(), z)
    dobj = dobj.item()
    gap = pobj - dobj
    return pobj1, pobj2, dobj, gap


def update_step(DDT, DDTz, Dy, lambda_p, z, w, mu1, mu2, f1, f2, mu, mu_inc, step, gap, m, alpha, beta, maxlsiter):
    """Update Newton's step for z, mu1, mu2, f1, f2"""

    # Update scheme for mu

    if step >= 0.2:
        mu_inc = max(2 * m * mu / gap, 1.2 * mu_inc)
    mu_inc_inv = 1 / mu_inc
    # step size of dual variable for equality
    rz = DDTz - w

    S = DDT - np.diag((mu1 / f1 + mu2 / f2).flatten())

    r = -DDTz + Dy + mu_inc_inv / f1 - mu_inc_inv / f2
    dz = np.linalg.solve(S, r)

    # step size for the dual variables formulated from constraints
    dmu1 = -(mu1 + (mu_inc_inv + dz * mu1) / f1)
    dmu2 = -(mu2 + (mu_inc_inv - dz * mu2) / f2)

    # residual of dual variables
    residual = np.vstack((rz, -mu1 * f1 - mu_inc_inv, -mu2 * f2 - mu_inc_inv))

    negIdx1 = np.where(dmu1 < 0)[0]
    negIdx2 = np.where(dmu2 < 0)[0]

    step = 1
    if len(negIdx1) > 0:
        step = min(step, 0.99 * np.min(-mu1[negIdx1] / dmu1[negIdx1]))
    if len(negIdx2) > 0:
        step = min(step, 0.99 * np.min(-mu2[negIdx2] / dmu2[negIdx2]))

    # Backtracking style line search, parameterized by alpha and beta
    for liter in range(maxlsiter):

        # update params within linesearch
        newz = z + step * dz
        newmu1 = mu1 + step * dmu1
        newmu2 = mu2 + step * dmu2
        newf1 = newz - lambda_p
        newf2 = -newz - lambda_p

        newResidual = np.vstack(
            (np.dot(DDT, newz) - Dy + newmu1 - newmu2, -newmu1 * newf1 - 1 / mu_inc, -newmu2 * newf2 - 1 / mu_inc)
        )

        # break out if actual reduction meets expected via norm of residual
        if max(np.max(newf1), np.max(newf2)) < 0 and (
            np.linalg.norm(newResidual) <= (1 - alpha * step) * np.linalg.norm(residual)
        ):
            break

        # must not return step otherwise converges to zero
        step *= beta

    return newz, newmu1, newmu2, newf1, newf2
