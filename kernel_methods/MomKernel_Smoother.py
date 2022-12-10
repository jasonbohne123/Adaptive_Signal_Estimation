import numpy as np
import sys
import math
path = '/home/jbohn/jupyter/personal/'
sys.path.append(f'{path}Adaptive_Signal_Estimation_Private/kernel_methods/')
from Kernel_Smoother import KernelSmoother

# TO:DO choose optimal block size based on out of sample set in cross validation

class MomKernelSmoother(KernelSmoother):
    """ Median of Means Kernel Smoother Class
    """

    def __init__(self, kernel_smoother, N=None):
        self.prior = kernel_smoother.prior
        self.index = kernel_smoother.index
        self.bandwidth_style = kernel_smoother.bandwidth_style

        # if N is not specified, then we will cross validate across a grid of block sizes between trivial and 2% of the length of the prior
        if N is None:
            grid = np.unique(np.floor(np.linspace(1, len(self.index)/50, 10))).astype(int)
            sorted_cv = self.cv_block_size(self.prior, grid=grid, verbose=True)
            self.N = sorted_cv[0]
            print(f"Optimal block size is {self.N}")
        else:
            self.N = N

    def partition_blocks(self, N,oos_indices=None):
        """ Partition prior into N blocks
        """

        if oos_indices is not None:
            all_indices = np.arange(len(self.index))
        else:
            all_indices =np.setdiff1d(np.arange(len(self.index)),oos_indices).sort()
        
        partition_indices = np.array_split(all_indices, N)

        blocked_prior = []
        blocked_index = []
        for i in range(N):
            blocked_prior.append(
                self.prior[partition_indices[i][0]:partition_indices[i][-1]])
            blocked_index.append(
                self.index[partition_indices[i][0]:partition_indices[i][-1]])

        return blocked_prior, blocked_index

    def fit_mom_kde(self, N=None,oos_indices=None):
        """ Applies robust kernel density estimation median of means 
        """
        if N is None:
            N = self.N

        # partition prior into N blocks, if testing oos pass to block partition
        block_prior, block_index = self.partition_blocks(N,oos_indices)

        # fit kde on each block
        kde_estimates = np.zeros((N, len(self.index)))

        for i in range(N):
            # determine the smooth values for each block
            blocked_kernel = KernelSmoother(
                block_prior[i], block_index[i], bandwidth_style=self.bandwidth_style)
            fitted_kernel_matrix = blocked_kernel.fit()

            # evaluate across range ( which includes extrapolation)
            for j in range(len(self.index)):
                kernel_eval = blocked_kernel.evaluate_kernel(self.index[j])

                kde_estimates[i, j] = kernel_eval

        # take median of estimates across each block ignoring nans
        kde = np.nanmedian(kde_estimates, axis=0)

        return kde 

    def cv_block_size(self, true, grid, verbose=False):
        """ Cross validates block size for kernel density estimation"""
        results = {}

        for n_i in grid:
            
            # reserve 25% of data for oos
            oos_indices= np.random.choice(np.arange(len(self.index)), size=math.floor(len(self.index)/4), replace=False)
            
            # compare true to kde of blocks n_i
            kde = self.fit_mom_kde(n_i,oos_indices)

            kde_oos = np.zeros(len(oos_indices))
            # predict oos
            for i in oos_indices:
                # need to be able to evaluate arbitrary points for median of means kernel
                continue

            mse = np.round(np.sum((true[oos_indices]-kde_oos)**2), 2)

            results[n_i] = [mse]
            if verbose:
                print(f" OOS MSE for {n_i} blocks is {mse}")

        return sorted(results.items(), key=lambda x: x[1])[0]
