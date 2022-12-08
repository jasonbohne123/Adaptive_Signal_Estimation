import numpy as np
from scipy.stats import iqr

class KernelSmoother:
    """ Kernel Smoother Class
    """
    def __init__(self, prior, index, bandwidth_style):
        self.prior = prior
        self.index = index
        self.bandwidth_style = bandwidth_style


    def compute_kernel(self,x_0,x_i,bandwidth):
        """ Given two points x_0 and x_i; compute the gaussian kernel utilizing euclidean distance 
        """
        if bandwidth==0:
            return 0
        scale=abs((x_0-x_i)/bandwidth)
        
        weight= np.exp(-(scale**2))
        
        return weight

    def fit(self):
        """
        Fits a kernel smoothing estimator on a prior series
        """

        if len(self.prior)!=len(self.index):
            print("Mismatched Series")
            return None
        
        kernel_matrix=np.zeros(len(self.index))
        
        # optimal bandwidth selection are for gaussian kernels 
        if self.bandwidth_style==0:
            bw=0.9*min(np.std(self.index),iqr(self.index)/1.35)/(len(self.index)**0.2) 
        else:
            bw=1.06*np.std(self.index) / (len(self.index)**0.2)


        for i in range(0,len(self.index)):
            denominator=0
            weight=0

            for j in range(0,len(self.index)):
                kernel=self.compute_kernel(self.index[i],self.index[j],bw) 
                
                # normalize kernel matrix to unit density
                denominator+=kernel
                weight+=kernel*self.prior[j]
            
            kernel_matrix[i]=weight/denominator

        self.fitted_kernel_matrix=kernel_matrix
        return 

    def smooth_series(self):
        """ Smooths the series using the kernel smoothing estimator 
        """

        
        self.smooth_prior=self.fitted_kernel_matrix
        
        return


class MomKernelSmoother(KernelSmoother):
    """ Median of Means Kernel Smoother Class
    """
    def __init__(self,kernel_smoother):
        self.prior=kernel_smoother.prior
        self.index=kernel_smoother.index
        self.bandwidth_style=kernel_smoother.bandwidth_style
        self.fitted_kernel_matrix=kernel_smoother.fitted_kernel_matrix


    def partition_blocks(self,prior,index,N):
        """ Partition prior into N blocks
        """
        all_indices=np.arange(len(index))
        partition_indices=np.array_split(all_indices,N)

        blocked_prior=[]
        blocked_index=[]
        for i in range(N):
            blocked_prior.append(prior[partition_indices[i]])
            blocked_index.append(index[partition_indices[i]])
            
        return blocked_prior,blocked_index


    def mom_kde(self,prior,index,N,bandwidth):
        """ Applies robust kernel density estimation median of means 
        """
        kde_list=[]
        
        # partition prior into N blocks
        block_prior,block_index=self.partition_blocks(prior,index,N)

        # fit kde on each block
        kde_estimates=np.empty((N,len(index)))
        
        for i in range(N):
            # determine the smooth values for each block
            kde_estimates[i]=smooth_series(block_prior[i],block_index[i])[0]

        # take median of estimates across each block
        kde=np.median(kde_estimates,axis=0)

        # rescale to unit density
        kde=kde/np.sum(kde)

        return kde 
        

    def cv_block_size(self,true,prior,index,bw,grid,verbose=False):
        """ Cross validates block size for kernel density estimation"""
        results={}

        for n_i in grid:
            kde=self.mom_kde(prior,index,n_i,bw)
            mse=np.round(np.sum((true-kde)**2),2)
            
            results[n_i]=[mse]
            if verbose:
                print(f" MSE for {n_i} blocks is {mse}")

        return sorted(results.items(), key=lambda x:x[1])[0]