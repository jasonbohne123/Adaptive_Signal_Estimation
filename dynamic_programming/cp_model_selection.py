from dp_recursion import map_intervals,best_fit_polynomial
import numpy as np

def generalized_cross_validation(optimal_indices,Y,order=1):
    """ Determines optimal number of changepoints based on generalized cross validation """

    mse_dict={}
    gcv_dict={}
    
    # confirm below params are optimal
    c=3
    r=order+2

    for k_i, cps in optimal_indices.items():

        # pad cps
        cps=np.unique(np.concatenate([[0],cps,[len(Y)]]))

        fixed_intervals=map_intervals(Y,cps)
      
        

        K=len(cps)
        M=r+c*K

        fixed_mse=0
        # compute the sum of squared errors of best fitted polynomial each interval
        
        for inter in list(fixed_intervals.values()):
            mse=best_fit_polynomial(Y,inter)
            fixed_mse+=mse
        
        mse_dict[k_i]=fixed_mse
        gcv_dict[k_i]=fixed_mse/(1-(M/len(Y))**2)

    sorted_mse=sorted(mse_dict.items(),key=lambda x: x[1])
    sorted_gcv=sorted(gcv_dict.items(),key=lambda x: x[1])
    return sorted_mse, sorted_gcv