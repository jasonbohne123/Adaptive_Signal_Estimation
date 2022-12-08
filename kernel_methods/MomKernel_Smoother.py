import numpy as np
from KernelSmoother import KernelSmoother


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