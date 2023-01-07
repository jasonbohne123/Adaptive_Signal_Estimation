import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.opt_params import get_hyperparams


def adaptive_tf(y, D_=Difference_Matrix, lambda_p=1.0, k=2, verbose=True):
    """
    Adaptive trend filtering algorithm
    """

    hyperparams = get_hyperparams()
    alpha, beta, gamma, mu, mu_inc, maxiter, maxlsiter, tol = map(
        hyperparams.get, ["alpha", "beta", "gamma", "mu", "mu_inc", "maxiter", "maxlsiter", "tol"]
    )

    n = len(y)
    m = n - k

    if len(lambda_p) == 1:
        lambda_p = np.array(lambda_p) * np.ones((m, 1))

    D = D_.D
    DDT = D_.DDT
    DDT_inv = D_.DDT_inv

    Dy = np.dot(D, y)

    # init variables and objectives
    z = np.zeros((m, 1))
    mu1 = np.ones((m, 1))
    mu2 = np.ones((m, 1))
    mu_inc = 1e-10
    np.inf
    dobj = 0
    step = np.inf
    f1 = z[0] - lambda_p
    f2 = -z[0] - lambda_p

    # main loop of iteration
    for iters in range(maxiter + 1):

        DTz, DDTz, w = prep_matrices(D, Dy, z, mu1, mu2)

        # compute objectives
        pobj1, pobj2, dobj, gap = compute_objective(DDT_inv, Dy, DTz, DDTz, z, w, mu1, mu2, lambda_p)

        print("Duality Gap is {}".format(gap))

        if verbose:
            if iters % 5 == 0:
                print(f"pobj1: {pobj1}, pobj2: {pobj2}, dobj: {dobj}, gap: {gap}")

        # if duality gap becomes negative
        if gap < 0:
            status = "negative duality gap"
            if verbose:
                print(status)
                print(f"pobj1: {pobj1}, pobj2: {pobj2}, dobj: {dobj}, gap: {gap}")
            x = y - np.dot(D.transpose(), z)
            return x, status, gap

        # if duality gap is small enough
        if gap <= tol:
            status = "solved"
            print(status)
            print(f"pobj1: {pobj1}, pobj2: {pobj2}, dobj: {dobj}, gap: {gap}")
            x = y - np.dot(D.transpose(), z)
            return x, status, gap

        # update step
        newz, newmu1, newmu2, newf1, newf2 = update_step(
            DDT, DDTz, Dy, lambda_p, z, w, mu1, mu2, f1, f2, mu, mu_inc, step, gap, m, alpha, beta, maxlsiter
        )

        # adaptive stepsize of mu with ratio gamma
        newmu1, newmu2 = adaptive_step_size(pobj1, pobj2, newmu1, newmu2, gamma)

        z = newz
        mu1 = newmu1
        mu2 = newmu2
        f1 = newf1
        f2 = newf2

    x = y - np.dot(D.transpose(), z)
    if iters >= maxiter:
        status = "maxiter exceeded"
        print(status)
        return x, status, gap


def prep_matrices(D, Dy, z, mu, mu2):
    """Prep matrices for objective computation"""

    DTz = np.dot(D.T, z)
    DDTz = np.dot(D, DTz)
    w = Dy - (mu - mu2)
    return DTz, DDTz, w


def compute_objective(DDT_inv, Dy, DTz, DDTz, z, w, mu1, mu2, lambda_p):
    """Computes Primal and Dual objectives and duality gap"""
    pobj1 = 0.5 * np.dot(w.T, (np.dot(DDT_inv, w))) + np.sum(np.dot(lambda_p.T, (mu1 + mu2)))
    pobj2 = 0.5 * np.dot(DTz.transpose(), DTz) + np.sum(np.dot(lambda_p.T, abs(Dy - DDTz)))
    pobj = min(pobj1, pobj2)
    dobj = -0.5 * np.dot(DTz.transpose(), DTz) + np.dot(Dy.transpose(), z)
    gap = pobj - dobj
    return pobj1, pobj2, dobj, gap


def update_step(DDT, DDTz, Dy, lambda_p, z, w, mu1, mu2, f1, f2, mu, mu_inc, step, gap, m, alpha, beta, maxlsiter):
    """Update step for z, mu1, mu2, f1, f2"""

    # Update scheme for mu
    if step >= 0.2:
        mu_inc = max(2 * m * mu / gap, 1.2 * mu_inc)

    # step size of dual variable for equality
    rz = DDTz - w

    S = DDT - np.diag((mu1 / f1 + mu2 / f2).flatten())
    S_inv = np.linalg.inv(S)

    r = -DDTz + Dy + (1 / mu_inc) / f1 - (1 / mu_inc) / f2
    dz = np.dot(S_inv, r)

    # step size for the dual variables formulated from constraints
    dmu1 = -(mu1 + ((1 / mu_inc) + dz * mu1) / f1)
    dmu2 = -(mu2 + ((1 / mu_inc) - dz * mu2) / f2)

    # residual of dual variables
    resDual = rz
    resCent = np.vstack((-mu1 * f1 - 1 / mu_inc, -mu2 * f2 - 1 / mu_inc))
    residual = np.vstack((resDual, resCent))

    negIdx1 = dmu1 < 0
    negIdx2 = dmu2 < 0

    step = 1
    if negIdx1.any():
        step = min(step, 0.99 * min(-mu1[negIdx1] / dmu1[negIdx1]))
    if negIdx2.any():
        step = min(step, 0.99 * min(-mu2[negIdx2] / dmu2[negIdx2]))

    # Backtracking style line search, parameterized by alpha and beta
    for liter in range(maxlsiter):

        # update params within linesearch
        newz = z + step * dz
        newmu1 = mu1 + step * dmu1
        newmu2 = mu2 + step * dmu2
        newf1 = newz - lambda_p
        newf2 = -newz - lambda_p

        newResDual = np.dot(DDT, newz) - Dy + newmu1 - newmu2
        newResCent = np.vstack((-newmu1 * newf1 - 1 / mu_inc, -newmu2 * newf2 - 1 / mu_inc))
        newResidual = np.vstack((newResDual, newResCent))

        # break out if actual reduction meets expected via norm of residual
        if max(max(newf1), max(newf2)) < 0 and (
            np.linalg.norm(newResidual) <= (1 - alpha * step) * np.linalg.norm(residual)
        ):
            break

        # must not return step otherwise converges to zero
        step *= beta

    return newz, newmu1, newmu2, newf1, newf2


def adaptive_step_size(pobj1, pobj2, newmu1, newmu2, gamma):
    """Adaptive step size of mu with ratio gamma"""
    if 2 * pobj1 > pobj2:
        newmu1 = newmu1 / gamma
        newmu2 = newmu2 * gamma
    elif 2 * pobj2 > pobj1:
        newmu1 = newmu1 * gamma
        newmu2 = newmu2 * gamma
    else:
        pass
    return newmu1, newmu2
