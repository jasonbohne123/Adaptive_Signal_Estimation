import numpy as np
import sys
path='/home/jbohn/jupyter/personal/'
sys.path.append(f'{path}Adaptive_Signal_Estimation_Private/kernel_methods/')
from Kernel_Smoother import KernelSmoother


class MomKernelSmoother(KernelSmoother):
    """ Median of Means Kernel Smoother Class
    """
    def __init__(self,kernel_smoother,N=None):
        self.prior=kernel_smoother.prior
        self.index=kernel_smoother.index
        self.bandwidth_style=kernel_smoother.bandwidth_style
        self.fitted_kernel_matrix=kernel_smoother.fitted_kernel_matrix
        self.N=N
        if N is None:
            self.N=10
            # CV

    def partition_blocks(self,N):
        """ Partition prior into N blocks
        """
        all_indices=np.arange(len(self.index))
        partition_indices=np.array_split(all_indices,N)
        print(partition_indices)

        blocked_prior=[]
        blocked_index=[]
        for i in range(N):
            blocked_prior.append(self.prior[partition_indices[i][0]:partition_indices[i][-1]])
            blocked_index.append(self.index[partition_indices[i][0]:partition_indices[i][-1]])
            
        return blocked_prior,blocked_index


    def fit_mom_kde(self,N=None):
        """ Applies robust kernel density estimation median of means 
        """
        if N is None:
            N=self.N
        # partition prior into N blocks
        block_prior,block_index=self.partition_blocks(N)

        # fit kde on each block
        kde_estimates=np.empty((N,len(self.index)))
        
        for i in range(N):
            # determine the smooth values for each block
            blocked_kernel=KernelSmoother(block_prior[i],block_index[i],bandwidth_style=self.bandwidth_style)
            blocked_kernel.fit()
            blocked_kernel.smooth_series()
            for j in range(len(self.index)):
                kde_estimates[i,j]=blocked_kernel.evaluate_kernel(self.index[j])
       
       
        kde_estimates=kde_estimates[kde_estimates!=None]
        # take median of estimates across each block
        kde=np.median(kde_estimates,axis=0)

        # rescale to unit density
        kde=kde/np.sum(kde)

        self.smooth_prior=kde

        return 
        

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