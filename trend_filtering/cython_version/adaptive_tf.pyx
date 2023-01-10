### Cython Implementation of Adaptive Trend Filtering
import Cython
cimport numpy as np

# necessary for calling numpy API
np.import_array()


from trend_filtering.opt_params import get_hyperparams

def adaptive_trend_filter(y,t=None,lambda=1.0,k=2,verbose=True):

    # define variable types 
    cdef float gamma,alpha,beta,mu,tol
    cdef int max_iter,maxlsiter

    # allow gil to be released for hyperparameter fetching 
    with gil:
        opt_params = get_hyperparams(y,t,lambda,k,verbose)
        gamma=opt_params["gamma"]
        alpha=opt_params["alpha"]
        beta=opt_params["beta"]
        mu=opt_params["mu"]
        max_iter=opt_params["max_iter"]
        maxlsiter=opt_params["maxlsiter"]
        tol=opt_params["tol"]
    
    cdef int n,m

    n=y.shape[0]
    m=n-2

    # compute difference matrices and their inverses ; sparse algorithm
    D = Dmat(n, k).toarray()
    DDT = np.dot(D, D.transpose())
    DDT_inv = sparse_inversion(DDT)
    Dy = np.dot(D, y)
    





