import numpy as np

def l1tf_adaptive(y, lambda_p,k=2):
    
    '''
    y is observed signal, lambda_p is penalization value
    
    
    
    '''
    alpha     = 0.01
    beta      = 0.5
    mu        = 2
    maxiter   = 40
    maxlsiter = 20
    tol       = 1e-4 # tol for duality gap to terminate upon
    
    n = len(y)
    m = n-k
    
    if not isinstance(lambda_p, np.ndarray):
        lambda_p=lambda_p*np.ones((m,1))
     
    
    # Difference operators for algorithm
    D = Dmat(n, k) # kth order difference matrix
    DDT = np.dot(D, D.transpose()) 
    
    # initial difference guess
    Dy = np.dot(D, y) 
    
    z = np.zeros((m, 1)) # this is our dual variable 
    mu1 = np.ones((m, 1))
    mu2 = np.ones((m, 1))
    
    t = 1e-10
    pobj = np.inf # primal objective
    dobj = 0 # dual objective
    step = np.inf
    
    # initial values for f1, f2 , from vector lambda_p
    f1   =  z-lambda_p
    f2   = -z-lambda_p
    
    for iters in range(maxiter):
        
        # rhs of algorithm to implement 
        DTz = np.dot(z.transpose(), D).transpose()
        DDTz = np.dot(D, DTz)
        w = Dy - (mu1 -mu2)
     
        # how are the two primal objectives equivalent 
        pobj1 = 0.5* np.dot(w.transpose(), (np.dot(np.linalg.inv(DDT), w)))+np.sum(np.dot(lambda_p.T,(mu1+mu2)))
        pobj2 = 0.5* np.dot(DTz.transpose(),DTz) + np.sum(np.dot(lambda_p.T, abs(Dy-DDTz)))
        
       
        pobj = min(pobj1,pobj2)  #sensitive of shaping of arrays
        dobj = -0.5* np.dot(DTz.transpose(), DTz)+ np.dot(Dy.transpose(), z) # dual objective
        
        gap = pobj - dobj #duality gap
        
        if gap <= tol:
            status = "solved"
            print(status)
            x = y - np.dot(D.transpose(), z) # solution
            return x, status,gap
        
        # update scheme for the step within primal-dual interior point method , which is a fist order approximation
        # note the step size for the search direction is a function of the residuals for both objectives as the system is coupled 
        if (step >= 0.2):
            t =max(2*m*mu/gap, 1.2*t)
        
        rz = DDTz - w
        S = DDT - np.diag((mu1/f1 + mu2/f2).flatten()) # Jacobian J_1 J_2
        r = -DDTz + Dy + (1/t)/f1 - (1/t)/f2
        dz = np.dot(np.linalg.inv(S), r) # step size for dual variable representing equality 
        
        # step size for the dual variables formulated from constraints 
        dmu1 = -(mu1+((1/t)+dz * mu1)/f1)
        dmu2 = -(mu2+((1/t)-dz * mu2)/f2)
        
        resDual = rz # residual of dual
        resCent = np.vstack((-mu1 * f1-1/t, -mu2 * f2-1/t))
        residual = np.vstack((resDual, resCent))
        
        negIdx1 = (dmu1 < 0)
        negIdx2 = (dmu2 < 0)
        
        
        step = 1
        if negIdx1.any():
            step = min(step, 0.99*min(-mu1[negIdx1]/dmu1[negIdx1]))
        if negIdx2.any():
            step = min(step, 0.99*min(-mu2[negIdx2]/dmu2[negIdx2]))
        
        # Backtracking style line search, parameterized by alpha and beta 
        for liter in range(maxlsiter):
            
            newz    =  z  + step*dz
            newmu1  =  mu1 + step*dmu1
            newmu2  =  mu2 + step*dmu2
            newf1   =  newz - lambda_p
            newf2   = -newz - lambda_p
            
            newResDual  = np.dot(DDT, newz) - Dy + newmu1 - newmu2
            newResCent  = np.vstack((-newmu1*newf1-1/t, -newmu2*newf2-1/t))
            newResidual = np.vstack((newResDual, newResCent))
            
            # break out if actual reduction meets expected via norm of residual
            if max(max(newf1),max(newf2)) < 0 and (np.linalg.norm(newResidual) <= (1-alpha*step)*np.linalg.norm(residual)):
                break
            
            step *= beta
        
        z  = newz
        mu1 = newmu1
        mu2 = newmu2
        f1 = newf1
        f2 = newf2
        
    x = y - np.dot(D.transpose(), z)
    if iters >= maxiter:
        status = 'maxiter exceeded'
        print(status)
        return x, status,None
    
def Dmat(n,k):
    """
        function reform a matrix for assets with order
        
        Parameters
        ----------
        n : int
        k: int
        
        Returns
        -------
        D : Array
    """
    if k == 0:
        D = np.eye(n)
    elif k == 1:
        D = np.eye(n-1,n)
        for i in range(n-1):
            D[i, i+1] = -1
    else:
        D = Dmat(n,1)
        for i in range(k-1):
            Dn = Dmat(n-i-1, 1)
            D = np.dot(Dn, D)
    return D