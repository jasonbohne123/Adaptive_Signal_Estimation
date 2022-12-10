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
        if N is None:
            sorted_cv=self.cv_block_size(self.prior,grid=np.floor(np.linspace(2,len(self.index)/10,10)).astype(int),verbose=True)
            self.N=sorted_cv[0]
            print(f"Optimal block size is {self.N}")
        else:
            self.N=N
        


    def partition_blocks(self,N):
        """ Partition prior into N blocks
        """
        all_indices=np.arange(len(self.index))
        partition_indices=np.array_split(all_indices,N)
      
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
            fitted_kernel_matrix=blocked_kernel.fit()
            smooth=blocked_kernel.smooth_series(fitted_kernel_matrix)
            for j in range(len(self.index)):
                kde_estimates[i,j]=blocked_kernel.evaluate_kernel(self.index[j])
       
       
        kde_estimates=kde_estimates[np.isnan(kde_estimates)==False]
        # take median of estimates across each block
        kde=np.median(kde_estimates,axis=0)

        # rescale to unit density
        kde=kde/np.sum(kde)

        return kde 
        
    # TO:DO: why isn't CV working for different block sizes?
    def cv_block_size(self,true,grid,verbose=False):
        """ Cross validates block size for kernel density estimation"""
        results={}

        for n_i in grid:
            kde=self.fit_mom_kde(N=n_i)
            mse=np.round(np.sum((true-kde)**2),2)
            
            results[n_i]=[mse]
            if verbose:
                print(f" MSE for {n_i} blocks is {mse}")

        return sorted(results.items(), key=lambda x:x[1])[0]