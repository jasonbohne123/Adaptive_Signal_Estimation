import numpy as np
from scipy.sparse import dia_matrix
from matrix_algorithms.difference_matrix import Difference_Matrix
from trend_filtering.opt_params import get_hyperparams


def adjust_penalty_time(lambda_p, times, k, verbose):
    """ Adjusts penalty by difference in time between observations
    """

    if times is None:
        if verbose:
            print("No time information provided, using default penalty")
        return lambda_p

    if k != 2:
        if verbose:
            print("Time information provided, but k is not 2, using default penalty")
        return lambda_p

    n = len(times)
    t_diff = np.diff(times)

    # forms three tridiagonal of time penalties in linear trend filtering
    t_diff1 =np.fmax(0.01, np.pad(t_diff, (0, 1), "constant", constant_values=t_diff[-1]))
    t_diff3 =np.fmax(0.01, np.pad(
        t_diff, (0, 1), "constant", constant_values=t_diff[-1]
    ))
    t_diff2 = t_diff1 + t_diff3

    a = t_diff1 / t_diff3
    b = -t_diff2 / (t_diff1 * t_diff3)
    c = t_diff3 / t_diff1

    # constructs a banded matrix of time differences
    T = dia_matrix(
        (np.vstack([a, b, c]), [0, 1, 2]), shape=(n - 2, n-2)
    ).toarray()

    # Scales our penalty by the time differences
    lambda_p = np.dot(lambda_p.T, abs(T)).reshape(-1, 1)

    return lambda_p


def adaptive_tf(y, t=None, lambda_p=1.0, k=2, verbose=True):
    """
    Adaptive trend filtering algorithm
    """

    hyperparams = get_hyperparams()
    gamma = hyperparams["gamma"]
    alpha = hyperparams["alpha"]
    beta = hyperparams["beta"]
    mu = hyperparams["mu"]
    maxiter = hyperparams["maxiter"]
    maxlsiter = hyperparams["maxlsiter"]
    tol = hyperparams["tol"]

    n = len(y)
    m = n - k

    if len(lambda_p) == 1:
        lambda_p = np.array(lambda_p) * np.ones((m, 1))

    lambda_p = adjust_penalty_time(lambda_p, t, k, verbose)

    # construct difference matrix
    diff_mat = Difference_Matrix(n, k)
    D = diff_mat.D
    DDT = diff_mat.DDT
    DDT_inv=diff_mat.DDT_inv

    Dy = np.dot(D, y)

    # init variables and objectives
    z = np.zeros((m, 1))
    mu1 = np.ones((m, 1))
    mu2 = np.ones((m, 1))
    mu_inc = 1e-10
    pobj = np.inf
    dobj = 0
    step = np.inf
    f1 = z - lambda_p
    f2 = -z - lambda_p

    # main loop of iteration
    for iters in range(maxiter + 1):

        DTz = np.dot(z.transpose(), D).transpose()
        DDTz = np.dot(D, DTz)
        w = Dy - (mu1 - mu2)

        # compute primal and dual objective values
        pobj1 = 0.5 * np.dot(w.T, (np.dot(DDT_inv, w))) + np.sum(
            np.dot(lambda_p.T, (mu1 + mu2))
        )
        pobj2 = 0.5 * np.dot(DTz.transpose(), DTz) + np.sum(
            np.dot(lambda_p.T, abs(Dy - DDTz))
        )
        pobj = min(pobj1, pobj2)
        dobj = -0.5 * np.dot(DTz.transpose(), DTz) + np.dot(
            Dy.transpose(), z
        )
        gap = pobj - dobj
        print("Duality Gap is {}".format(gap))

        if verbose:
            if iters % 5 == 0:
                print(
                    f"pobj1: {pobj1}, pobj2: {pobj2}, dobj: {dobj}, gap: {gap}")

        # if duality gap becomes negative
        if gap < 0:
            status = "negative duality gap"
            if verbose:
                print(status)
                print(
                    f"pobj1: {pobj1}, pobj2: {pobj2}, dobj: {dobj}, gap: {gap}")
            x = y - np.dot(D.transpose(), z)
            return x, status, gap

        # if duality gap is small enough
        if gap <= tol:
            status = "solved"
            print(status)
            print(f"pobj1: {pobj1}, pobj2: {pobj2}, dobj: {dobj}, gap: {gap}")
            x = y - np.dot(D.transpose(), z)
            return x, status, gap

        # Update scheme for mu
        if step >= 0.2:
            mu_inc = max(2 * m * mu / gap, 1.2 * mu_inc)

        # step size of dual variable for equality
        rz = DDTz - w
        S = DDT - np.diag((mu1 / f1 + mu2 / f2).flatten())
        r = -DDTz + Dy + (1 / mu_inc) / f1 - (1 / mu_inc) / f2
        dz = np.dot(
            np.linalg.inv(S), r
        )

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
            newResCent = np.vstack(
                (-newmu1 * newf1 - 1 / mu_inc, -newmu2 * newf2 - 1 / mu_inc))
            newResidual = np.vstack((newResDual, newResCent))

            # break out if actual reduction meets expected via norm of residual
            if max(max(newf1), max(newf2)) < 0 and (
                np.linalg.norm(newResidual)
                <= (1 - alpha * step) * np.linalg.norm(residual)
            ):
                break

            step *= beta

        # adaptive stepsize of mu with ratio gamma
        if 2*pobj1 > pobj2:
            newmu1 = newmu1/gamma
            newmu2 = newmu2*gamma
        elif 2*pobj2 > pobj1:
            newmu1 = newmu1*gamma
            newmu2 = newmu2*gamma
        else:
            pass
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

def cv_tf_penalty(y,grid,t=None,verbose=True):
    """Cross Validation for constant TF penalty parameter lambda_p"""
    

    best_gap=np.inf
    best_lambda=None
    for lambda_i in grid:
        x,status,gap=adaptive_tf(y,t,[lambda_i],verbose=verbose)

        if gap<best_gap:
            best_gap=gap
            best_lambda=lambda_i
        
    return np.array([best_lambda]),np.array([best_gap])