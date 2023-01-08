import numpy as np
from numba import njit
from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.opt_params import get_hyperparams
from matrix_algorithms.woodbury_inversion import woodbury_matrix_inversion
from matrix_algorithms.sherman_morrison import sherman_morrison_recursion

###### Numba Integration
# Utilize @njit decorator to compile functions to machine code
# Blocks default to object mode
# First compilation takes longer to cache the compilation of the code
# Utilies Intel's ICC compiler through Numba 

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
    
    pobj=np.inf
    dobj = 0
    step = np.inf
    f1 = z[0] - lambda_p
    f2 = -z[0] - lambda_p

    # main loop of iteration
    for iters in range(maxiter + 1):

        DTz, DDTz, w = prep_matrices(D, Dy, z, mu1, mu2)

        # compute objectives
        pobj1, pobj2, dobj, gap = compute_objective(DDT_inv, Dy, DTz, DDTz, z, w, mu1, mu2, lambda_p)

        
        # if duality gap becomes negative
        if gap < 0:
            status = "negative duality gap"
            x = y - np.dot(D.transpose(), z)
            return {"sol": None, "status": status, "gap": -1}

        # if duality gap is small enough
        if gap <= tol:
            status = "solved"
            x = y - np.dot(D.transpose(), z)
            return {"sol": x, "status": status, "gap": gap}

        # update step
        newz, newmu1, newmu2, newf1, newf2 = update_step(
            DDT, DDTz, Dy, DDT_inv, lambda_p, z, w, mu1, mu2, f1, f2, mu, mu_inc, step, gap, m, alpha, beta, maxlsiter
        )

        # adaptive stepsize of mu with ratio gamma
        newmu1, newmu2 = adaptive_step_size(pobj1, pobj2, newmu1, newmu2, gamma)

        z = newz
        mu1 = newmu1
        mu2 = newmu2
        f1 = newf1
        f2 = newf2

    status = "maxiter exceeded"
    return {"sol": None, "status": status, "gap": -1}

@njit(fastmath=True, cache=True)
def prep_matrices(D, Dy, z, mu, mu2):
    """Prep matrices for objective computation"""

    DTz = np.dot(D.T, z)
    DDTz = np.dot(D, DTz)
    w = Dy - (mu - mu2)
    return DTz, DDTz, w

@njit(fastmath=True, cache=True)
def compute_objective(DDT_inv, Dy, DTz, DDTz, z, w, mu1, mu2, lambda_p):
    """Computes Primal and Dual objectives and duality gap"""
    pobj1 = 0.5 * np.dot(w.T, (np.dot(DDT_inv, w))) + np.sum(np.dot(lambda_p.T, (mu1 + mu2)))
    pobj2 = 0.5 * np.dot(DTz.transpose(), DTz) + np.sum(np.dot(lambda_p.T, np.abs(Dy - DDTz)))
    pobj1=pobj1.item()
    pobj2=pobj2.item()
    pobj = min(pobj1, pobj2)
    dobj = -0.5 * np.dot(DTz.transpose(), DTz) + np.dot(Dy.transpose(), z)
    dobj=dobj.item()
    gap = pobj - dobj
    return pobj1, pobj2, dobj, gap

@njit(fastmath=True, cache=True)
def update_step(DDT, DDTz, Dy,DDT_inv, lambda_p, z, w, mu1, mu2, f1, f2, mu, mu_inc, step, gap, m, alpha, beta, maxlsiter):
    """Update step for z, mu1, mu2, f1, f2"""

    # Update scheme for mu

    if step >= 0.2:
        mu_inc = max(2 * m * mu / gap, 1.2 * mu_inc)
    mu_inc_inv = 1 / mu_inc
    # step size of dual variable for equality
    rz = DDTz - w

    S = DDT - np.diag((mu1 / f1 + mu2 / f2).flatten())
    S_inv = np.linalg.inv(S)

    #S_inv=woodbury_matrix_inversion(-(mu1 / f1 + mu2 / f2), DDT_inv,step=20)

    r = -DDTz + Dy + mu_inc_inv / f1 - mu_inc_inv / f2
    dz = np.dot(S_inv, r)

    # step size for the dual variables formulated from constraints
    dmu1 = -(mu1 + (mu_inc_inv + dz * mu1) / f1)
    dmu2 = -(mu2 + (mu_inc_inv - dz * mu2) / f2)

    # residual of dual variables
    residual = np.vstack((rz,-mu1 * f1 - mu_inc_inv, -mu2 * f2 - mu_inc_inv))
    

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

        newResidual = np.vstack(( np.dot(DDT, newz) - Dy + newmu1 - newmu2,-newmu1 * newf1 - 1 / mu_inc, -newmu2 * newf2 - 1 / mu_inc))
        

        # break out if actual reduction meets expected via norm of residual
        if max(np.max(newf1), np.max(newf2)) < 0 and (
            np.linalg.norm(newResidual) <= (1 - alpha * step) * np.linalg.norm(residual)
        ):
            break

        # must not return step otherwise converges to zero
        step *= beta

    return newz, newmu1, newmu2, newf1, newf2

@njit(fastmath=True, cache=True,nogil=True)
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
