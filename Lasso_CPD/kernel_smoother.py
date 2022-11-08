import numpy as np
import pandas as pd
from scipy.sparse import dia_matrix, csc_matrix
from scipy.sparse.linalg import inv
from scipy.interpolate import interp1d
from scipy.stats import iqr
import sys

def compute_kernel(x_0,x_i,bandwidth):
    """ Given two points x_0 and x_i; compute the gaussian kernel utilizing euclidean distance 
    """
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

    return (weighted_kernel/denominator)

def smooth_series(prior,index,bw_style=1):
    
    '''
    Kernel Smooths numpy array using the gaussian kerenl
    
    i). Applies Kernel Smoothing in time based on prior 
    ii). Utilizes Scotts and Silvermans Bandwidth Selection Methods
        
    '''
    if len(prior)!=len(index):
        print("Mismatched Series")
        return None
    
    smooth=[]
    if bw_style==0:
        bw=0.9*min(np.std(index),iqr(index)/1.35)/(len(index)**0.2) # standard deviation
    else:
        bw=1.06*np.std(index) / (len(index)**0.2)
    
    
    for i in range(0,len(prior)):
        smoothed_val=kernel_smooth(index[i],prior,index,bandwidth=bw)
        smooth.append(smoothed_val)
    
    return smooth,bw

def evaluate_kernel(y,smooth_prior,index,bandwidth):

    
    y_hat=kernel_smooth(y,smooth_prior,index,bandwidth)
    
    return y_hat