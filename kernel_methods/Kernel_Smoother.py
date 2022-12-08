import numpy as np
from scipy.stats import iqr

class KernelSmoother:
    """ Kernel Smoother Class
    """
    def __init__(self, prior, index, bandwidth_style):
        self.prior = prior
        self.index = index
        self.bandwidth_style = bandwidth_style
        self.optimal_bandwidth = None


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
        
        kernel_matrix=np.zeros((len(self.index),len(self.index)))
        
        # optimal bandwidth selection are for gaussian kernels 
        if self.bandwidth_style==0:
            bw=0.9*min(np.std(self.index),iqr(self.index)/1.35)/(len(self.index)**0.2) 
        else:
            bw=1.06*np.std(self.index) / (len(self.index)**0.2)

        self.optimal_bandwidth=bw


        for i in range(0,len(self.index)):
           

            for j in range(0,len(self.index)):
                kernel=self.compute_kernel(self.index[i],self.index[j],bandwidth=self.optimal_bandwidth) 
                kernel_matrix[i,j]=kernel

        self.fitted_kernel_matrix=kernel_matrix/np.sum(kernel_matrix,axis=1)
        return 

    def smooth_series(self):
        """ Smooths the series using the kernel smoothing estimator 
        """
        self.smooth_prior=self.fitted_kernel_matrix.dot(self.prior)
        
        return
    
    def evaluate_kernel(self,y_i):
        """ Evaluates the kernel at a given point y_i
        """

        if self.optimal_bandwidth is None:
            if self.bandwidth_style==0:
                bw=0.9*min(np.std(self.index),iqr(self.index)/1.35)/(len(self.index)**0.2)
            else:
                bw=1.06*np.std(self.index) / (len(self.index)**0.2)
            self.optimal_bandwidth=bw

        kernel_matrix=np.zeros((len(self.index)))
        for i in range(0,len(self.index)):
            kernel=self.compute_kernel(self.index[i],y_i,self.optimal_bandwidth) 
            kernel_matrix[i]=kernel
        

        y_hat=np.sum(kernel_matrix*self.prior)/np.sum(kernel_matrix)

        return y_hat
        