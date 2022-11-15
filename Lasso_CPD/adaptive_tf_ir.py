import numpy as np
from scipy.sparse import dia_matrix
from sparse_inv import sparse_inversion


def l1tf_adaptive_ir(y, t=None, lambda_p=None,k=2):

    """
    y is observed signal

    t= times at which observations occur ; defaults to irregularly spaced

    lambda_p =penalty to adaptively filter; defaults to array

    """
    gamma=0.5
    alpha=0.01
    beta=0.5
    mu=2
    maxiter = 500# adjust
    maxlsiter = 50
    tol = 1e-2  # adjust for tol

    n = len(y)
    m = n - k
    if not isinstance(lambda_p, np.ndarray):

        lambda_p = lambda_p * np.ones((m, 1))

    # Difference operators for algorithm
    D = Dmat(n, k).toarray()  

    if t is not None:

        """Account for irregularly spaced timestamps; right now k=2 supported"""

        if k == 2:
            t_diff = np.diff(t)

            t_diff1 = np.pad(t_diff, (0, 1), "constant", constant_values=t_diff[-1])
            t_diff3 = np.pad(
                t_diff, (0, 1), "constant", constant_values=t_diff[-1]
            )
            t_diff2 = t_diff1 + t_diff3  # error in composition

            a = t_diff1 / t_diff3
            b = -t_diff2 / (t_diff1 * t_diff3)
            c = t_diff3 #/ t_diff1

            # project onto matrix
            T = dia_matrix(
                (np.vstack([a, b, c]), [0, 1, 2]), shape=(n - 2, n-2)
            ).toarray()
            
            lambda_p=np.dot(lambda_p.T,abs(T)).reshape(-1,1) # factor in the absolute difference in time 
         
            
        else:
            print(f"K={k} not supported yet for uneven time discretization")

    DDT = np.dot(D, D.transpose())
    DDT_inv = sparse_inversion(DDT)
    
    Dy=np.dot(D,y)
    z = np.zeros((m, 1))  #dual variable
    mu1 = np.ones((m, 1))
    mu2 = np.ones((m, 1))

    t = 1e-10
    pobj = np.inf  # primal objective
    dobj = 0  # dual objective
    step = np.inf

    # initial values for f1, f2 , from vector lambda_p
    f1 = z - lambda_p
    f2 = -z - lambda_p

    for iters in range(maxiter + 1):
        # rhs of algorithm to implement
        DTz = np.dot(z.transpose(), D).transpose() ### need to adjust
        DDTz = np.dot(D, DTz)
        w = Dy - (mu1 - mu2)
       
        # primal and dual
        pobj1 = 0.5 * np.dot(w.T, (np.dot(DDT_inv, w))) + np.sum(
            np.dot(lambda_p.T, (mu1 + mu2))
        )
        pobj2 = 0.5 * np.dot(DTz.transpose(), DTz) + np.sum(
            np.dot(lambda_p.T, abs(Dy - DDTz))
        )
        
     
        pobj = min(pobj1, pobj2)  
        
        #dual of the dual
        dobj = -0.5 * np.dot(DTz.transpose(), DTz) + np.dot(
            Dy.transpose(), z
        ) 
    
        gap = pobj - dobj  # duality gap
        
        if iters%5==0:
            print(f"pobj1: {pobj1}, pobj2: {pobj2}, dobj: {dobj}, gap: {gap}")  
        if gap<0:
            status = "negative duality gap"
            print(status)
            print(f"pobj1: {pobj1}, pobj2: {pobj2}, dobj: {dobj}, gap: {gap}")
            guess= y - np.dot(D.transpose(), z)
            return guess, status, D
            
        if gap <= tol:
            status = "solved"
            print(status)
            print(f"pobj1: {pobj1}, pobj2: {pobj2}, dobj: {dobj}, gap: {gap}")
            x = y - np.dot(D.transpose(), z)  # solution
            return x, status, D

        # update scheme for the step within primal-dual interior point method , which is a fist order approximation
        # note the step size for the search direction is a function of the residuals for both objectives as the system is coupled
        # implemented adaptive stepsizes foor increased stability at rate gamma 
        if step >= 0.2:
            t = max(2 * m * mu / gap, 1.2 * t)

        rz = DDTz - w
        S = DDT - np.diag((mu1 / f1 + mu2 / f2).flatten())  # Jacobian J_1 J_2
        r = -DDTz + Dy + (1 / t) / f1 - (1 / t) / f2
        dz = np.dot(
            np.linalg.inv(S), r
        )  # step size for dual variable representing equality

        # step size for the dual variables formulated from constraints
        dmu1 = -(mu1 + ((1 / t) + dz * mu1) / f1)
        dmu2 = -(mu2 + ((1 / t) - dz * mu2) / f2)

        resDual = rz  # residual of dual
        resCent = np.vstack((-mu1 * f1 - 1 / t, -mu2 * f2 - 1 / t))
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
                
            newz = z + step * dz
            newmu1 = mu1 + step * dmu1
            newmu2 = mu2 + step * dmu2
            newf1 = newz - lambda_p
            newf2 = -newz - lambda_p

            newResDual = np.dot(DDT, newz) - Dy + newmu1 - newmu2
            newResCent = np.vstack((-newmu1 * newf1 - 1 / t, -newmu2 * newf2 - 1 / t))
            newResidual = np.vstack((newResDual, newResCent))

            # break out if actual reduction meets expected via norm of residual
            if max(max(newf1), max(newf2)) < 0 and (
                np.linalg.norm(newResidual)
                <= (1 - alpha * step) * np.linalg.norm(residual)
            ):
                break

            step *= beta
        
        # Adaptive stepsizes for updating our mu direction 
        # Ref. Adaptive primal-dual hybrid gradient methods
        #####################
        if 2*pobj1>pobj2:
            newmu1=newmu1/gamma
            newmu2=newmu2*gamma
        elif 2*pobj2>pobj1:
            newmu1=newmu1*gamma
            newmu2=newmu2*gamma
        else:
            pass
        #####################
        z = newz
        mu1 = newmu1
        mu2 = newmu2
        f1 = newf1
        f2 = newf2

    x = y - np.dot(D.transpose(), z)
    if iters >= maxiter:
        status = "maxiter exceeded"
        print(status)
        return x, status, D


def Dmat(n, k):
    """
    Optimized difference matrix computation using scipy sparse matrices using pascals 

    Parameters
    ----------
    n : int
    k: int

    Returns
    -------
    D : Array
    """

    def pascals(k):
        pas = [0, 1, 0]
        counter = k
        while counter > 0:
            pas.insert(0, 0)
            pas = [np.sum(pas[i : i + 2]) for i in range(0, len(pas))]
            counter -= 1
        return pas

    coeff = pascals(k)
    coeff = [i for i in coeff if i != 0]
    coeff = [coeff[i] if i % 2 == 0 else -coeff[i] for i in range(0, len(coeff))]

    if k == 0:
        D = dia_matrix((np.ones(n), 0), shape=(n - 2, n))
    elif k == 1:
        D = dia_matrix(
            (np.vstack([i * np.ones(n) for i in coeff]), range(0, k + 1)),
            shape=(n - 2, n),
        )
    else:
        D = dia_matrix(
            (np.vstack([i * np.ones(n) for i in coeff]), range(0, k + 1)),
            shape=(n - 2, n),
        )

    return D