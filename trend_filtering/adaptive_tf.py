import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.opt_params import get_hyperparams

# To-Do ; Refactor Time Weighted Difference Matrix
# To-Do ; Compile Matrix Inversion code with Numba
# Numba only supports 1-D array indexing


def adaptive_tf(y, D_: Difference_Matrix, lambda_p, k=2, verbose=True):
    """
    Adaptive trend filtering algorithm
    """

    hyperparams = get_hyperparams()
    gamma = hyperparams["gamma"]
    alpha = hyperparams["alpha"]
    beta = hyperparams["beta"]
    mu = hyperparams["mu"]
    mu_inc = hyperparams["mu_inc"]
    maxiter = hyperparams["maxiter"]
    maxlsiter = hyperparams["maxlsiter"]
    tol = hyperparams["tol"]

    n = len(y)
    m = n - k

    if len(lambda_p) == 1:
        lambda_p = np.array(lambda_p) * np.ones(m)

    D = D_.D
    DDT = D_.DDT
    DDT_inv = D_.DDT_inv

    Dy = np.dot(D, y)

    # init variables and objectives
    z = np.zeros(m)
    mu1 = np.ones(m)
    mu2 = np.ones(m)

    np.inf
    dobj = 0
    step = np.inf

    f1 = z - lambda_p
    f2 = -z - lambda_p

    alpha, beta, gamma, mu, mu_inc, maxiter, maxlsiter, tol = map(
        hyperparams.get, ["alpha", "beta", "gamma", "mu", "mu_inc", "maxiter", "maxlsiter", "tol"]
    )

    # main loop of iteration
    for iters in range(maxiter + 1):

        DTz, DDTz, w = prep_matrices(D, Dy, z, mu1, mu2)

        # compute the objective
        pobj1, pobj2, dobj, gap = compute_objective(Dy, DDT_inv, DTz, DDTz, z, w, lambda_p, mu1, mu2)

        print("Duality Gap is {}".format(gap))
        if verbose:
            if iters % 5 == 0:
                print(f"pobj: {min(pobj1,pobj2)} ,dobj: {dobj}, gap: {gap}")

        # if duality gap becomes negative
        if gap < 0:
            status = "duality gap negative"
            if verbose:
                print(status)
            return {
                "sol": None,
                "status": status,
                "gap": gap,
            }

        # if duality gap is small enough
        if gap <= tol:
            status = "solved"
            if verbose:
                print(status)
                print(f"pobj: {min(pobj1,pobj2)} ,dobj: {dobj}, gap: {gap}")
            x = y - np.dot(D.transpose(), z)
            return {
                "sol": x,
                "status": status,
                "gap": gap,
            }

        # invert S matrix
        S_inv = np.linalg.inv(DDT - np.diag((np.divide(mu1, f1) + np.divide(mu2, f2))))

        # our algorithm us much slower
        # woodbury_matrix_inversion(-(np.divide(mu1,f1) +np.divide(mu2,f2)), DDT_inv)

        # compute step direction and step size
        step, dz, dmu1, dmu2, residual = update_step_size(DDTz, S_inv, Dy, w, step, mu1, mu2, f1, f2, gap, mu, mu_inc)

        # Backtracking style line search, parameterized by alpha and beta
        newz, newmu1, newmu2, newf1, newf2, step = line_search(
            DDT, Dy, lambda_p, z, dz, mu1, mu2, dmu1, dmu2, step, residual, alpha, beta, mu_inc, maxlsiter
        )

        # adaptive step size for faster convergence
        newmu1, newmu2 = adaptive_step_size(pobj1, pobj2, newmu1, newmu2, gamma)

        # update variables
        z = newz
        mu1 = newmu1
        mu2 = newmu2
        f1 = newf1
        f2 = newf2

    if iters >= maxiter:
        status = "maxiter exceeded"
        if verbose:
            print(status)
        return {
            "sol": None,
            "status": status,
            "gap": gap,
        }


# @njit(nogil=True,cache=True)
def prep_matrices(D, Dy, z, mu1, mu2):
    """Prep matrices for objective computation"""

    DTz = np.dot(D.T, z)
    DDTz = np.dot(D, DTz)
    w = Dy - (mu1 - mu2)

    return DTz, DDTz, w


# @njit(nogil=True,cache=True)
def compute_objective(Dy, DDT_inv, DTz, DDTz, z, w, lambda_p, mu1, mu2):
    """Compute primal and dual objective values"""

    # refactor each 1-d array into row or column vector
    pobj1 = 0.5 * np.dot(w.reshape(1, -1), (np.dot(DDT_inv, w).reshape(-1, 1))) + np.sum(
        np.dot(lambda_p.reshape(1, -1), (mu1 + mu2).reshape(-1, 1))
    )
    pobj2 = 0.5 * np.dot(DTz.T, DTz) + np.sum(np.dot(lambda_p.reshape(1, -1), np.abs(Dy - DDTz).reshape(-1, 1)))

    pobj1 = pobj1.item()
    pobj2 = pobj2.item()
    pobj = min(pobj1, pobj2)

    dobj = -0.5 * np.dot(DTz.transpose(), DTz) + np.dot(Dy.transpose(), z)

    dobj = dobj.item()
    gap = pobj - dobj

    return pobj1, pobj2, dobj, gap


# @njit(nogil=True,cache=True)
def update_step_size(DDTz, S_inv, Dy, w, step, mu1, mu2, f1, f2, gap, mu, mu_inc):
    "Update step size and direction"

    m = len(f1)

    # Update scheme for mu
    if step >= 0.2:
        mu_inc = max(2 * m * mu / gap, 1.2 * mu_inc)

    # step size of dual variable for equality
    rz = DDTz - w
    r = -DDTz + Dy + (1 / mu_inc) / f1 - (1 / mu_inc) / f2
    dz = (np.dot(S_inv, r.reshape(-1, 1))).flatten()

    # step size for the dual variables formulated from constraints
    dmu1 = -(mu1 + ((1 / mu_inc) + np.dot(dz, mu1.T)) / f1)
    dmu2 = -(mu2 + ((1 / mu_inc) - np.dot(dz, mu2.T)) / f2)

    residual = np.vstack((rz, -mu1 * f1 - 1 / mu_inc, -mu2 * f2 - 1 / mu_inc))

    negIdx1 = np.asarray(dmu1 < 0).nonzero()
    negIdx2 = np.asarray(dmu2 < 0).nonzero()

    step = 1.0

    if len(negIdx1[0]) > 0:

        min_el = np.min(-mu1[negIdx1] / dmu1[negIdx1]).item()
        step = min(step, 0.99 * min_el)

    if len(negIdx2[0]) > 0:

        min_el = np.min(-mu2[negIdx2] / dmu2[negIdx2]).item()
        step = min(step, 0.99 * min_el)

    return step, dz, dmu1, dmu2, residual


# @njit(nogil=True,cache=True)
def line_search(DDT, Dy, lambda_p, z, dz, mu1, mu2, dmu1, dmu2, step, residual, alpha, beta, mu_inc, maxlsiter):
    "Backtracking line search"
    for liter in range(maxlsiter):

        # update params within linesearch
        newz = z + step * dz
        newmu1 = mu1 + step * dmu1
        newmu2 = mu2 + step * dmu2
        newf1 = newz - lambda_p
        newf2 = -newz - lambda_p

        newResDual = np.dot(DDT, newz) - Dy + newmu1 - newmu2
        newResidual = np.vstack((newResDual, -newmu1 * newf1 - 1 / mu_inc, -newmu2 * newf2 - 1 / mu_inc))

        # break out if actual reduction meets expected via norm of residual
        if max(np.max(newf1), np.max(newf2)) < 0 and (
            np.linalg.norm(newResidual) <= (1 - alpha * step) * np.linalg.norm(residual)
        ):
            break

        step *= beta
    return newz, newmu1, newmu2, newf1, newf2, step


# @njit(nogil=True,cache=True)
def adaptive_step_size(pobj1, pobj2, newmu1, newmu2, gamma):
    # adaptive stepsize of mu with ratio gamma
    if 2 * pobj1 > pobj2:
        newmu1 = newmu1 / gamma
        newmu2 = newmu2 * gamma
    elif 2 * pobj2 > pobj1:
        newmu1 = newmu1 * gamma
        newmu2 = newmu2 * gamma
    else:
        pass

    return newmu1, newmu2
