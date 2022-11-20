import numpy as np
from scipy.stats import iqr

def compute_kernel(x_0,x_i,bandwidth):
    """ Given two points x_0 and x_i; compute the gaussian kernel utilizing euclidean distance 
    """
    if bandwidth==0:
        return 0
    scale=abs((x_0-x_i)/bandwidth)
    
    weight= np.exp(-scale**2)
      
    return weight

def kernel_smooth(reference,prior,index,bandwidth):
    """ Applies Kernel Smoothing in time on a reference point given prior and index
    """
    
    weighted_kernel=0
    denominator=0
    counter=0
    for i in range(0,len(index)):
        kernel=compute_kernel(reference,index[i],bandwidth) 
        denominator+=kernel
        weighted_kernel+=kernel*prior[i]
        counter+=1
    
    if denominator==0:
        return reference

    return (weighted_kernel/denominator)

def smooth_series(prior,index,bw_style=1):
    """
    Kernel Smooths numpy array using the gaussian kerenl
    i). Applies Kernel Smoothing in time based on prior 
    ii). Utilizes Scotts and Silvermans Bandwidth Selection Methods
    """
    
    if len(prior)!=len(index):
        print("Mismatched Series")
        return None
    
    smooth=[]

    # optimal bandwidth selection are for gaussian kernels 
    if bw_style==0:
        bw=0.9*min(np.std(index),iqr(index)/1.35)/(len(index)**0.2) 
    else:
        bw=1.06*np.std(index) / (len(index)**0.2)
    
    
    for i in range(0,len(prior)):
        smoothed_val=kernel_smooth(index[i],prior,index,bandwidth=bw)
        smooth.append(smoothed_val)
    
    return np.array(smooth),bw

def evaluate_kernel(y,smooth_prior,index,bandwidth):
    """ Evaluates kernel arbitrarily given smoothed prior 
    """
    y_hat=kernel_smooth(y,smooth_prior,index,bandwidth)
    
    return y_hat


def mom_kde(prior,index,N,bandwidth):
    """ Applies robust kernel density estimation median of means 
    """
    kde_list=[]
    stepsize=np.floor(len(prior)/N).astype(int)+1

    for i in range(N):
        block_prior=prior[i*stepsize:(i+1)*stepsize]
        block_index=index[i*stepsize:(i+1)*stepsize]
        
        
        if len(block_prior)==0 or len(block_index)==0:
            continue
        
        # fit kde on block
        kde,bw=smooth_series(block_prior,block_index)
        kde_list.append(kde)
        
    kde_flatten=[kde_val for subkde in kde_list for kde_val in subkde]
    
    return np.array(kde_flatten)

def cv_block_size(true,prior,index,bw,grid,verbose=False):
    """ Cross validates block size for kernel density estimation"""
    results={}

    for n_i in grid:
        kde=mom_kde(prior,index,n_i,bw)
        mse=np.round(np.sum((true-kde)**2),2)
        
        results[n_i]=[mse]
        if verbose:
            print(f" MSE for {n_i} blocks is {mse}")

    return sorted(results.items(), key=lambda x:x[1])[0]