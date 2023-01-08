import numpy as np
from matrix_algorithms.difference_matrix import Difference_Matrix


class Time_Difference_Matrix(Difference_Matrix):
    """Time Weighted Difference Matrix for L1 Trend Filtering

        Inherits from Difference_Matrix so original matrices are accessible 
    
    """

    def __init__(self, matrix: Difference_Matrix, t=None):
        super().__init__(matrix.n, matrix.k, matrix.style)
        self.t= t
      
        # if time increments are not provided, assume they are unit spaced
        if t is None:
            t = np.arange(1, self.n + 1)

        # returns the time matrix T
        self.T_D = self.construct_time_matrix(t)
        self.T_DDT = self.T_D.dot( self.T_D.T)

                # determine the projected coefficients across diagonals
        T_DDT_diag_coeff = [self.T_DDT.diagonal(i)[0] for i in range(-self.k, self.k + 1)]

        self.T_DDT_diag = np.array([i * np.ones(self.n - 2) for i in T_DDT_diag_coeff])

        # need to fix this inversion 
        self.T_DDT_inv = np.asarray(self.invert(self.T_DDT_diag,style=self.style), order="C")

    def construct_time_matrix(self,t):
        """ Constructs time matrix T which is embedded in our difference matrix"""
        n=len(t)

        assert n==self.n, "Time increments must be same length as number of observations"

        assert self.k==2, "Time weighted difference matrix construction only works for k=2 atm "
        
        # reference time increment
        if t[0]!=1.0:
            t=[t[i]-t[0]+1.0 for i in range(0,n)]

        # iteratively construct the time matrix
        T=np.zeros((n-2,n))
        for i in range(0,n):
            for j in range(0,n-2):
                if i==j:
                    T[j,i]=t[j+1]-t[j]
                elif i==j+1:
                    T[j,i]=-((t[j+2]-t[j+1])+(t[j+1]-t[j]))
                elif i==j+2:
                    T[j,i]=t[j+2]-t[j+1]
                else:
                    T[j,i]=0.0
        return T
