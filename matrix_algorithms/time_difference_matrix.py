import numpy as np
from difference_matrix import Second_Order_Difference_Matrix


class Time_Difference_Matrix(Second_Order_Difference_Matrix):
    """Time Weighted Difference Matrix for L1 Trend Filtering"""

    def __init__(self, matrix: Second_Order_Difference_Matrix, time_increments=None):
        super().__init__(matrix.n, matrix.k, matrix.style)
        self.time_increments = time_increments

        if time_increments is None:
            time_increments = np.ones(self.n)

        self.T = self.construct_time_matrix(time_increments)

    def construct_time_matrix(self,time_increments):
        """ Constructs time matrix T which is embedded in our difference matrix"""
        n=len(time_increments)

        assert n==self.n, "Time increments must be same length as number of observations"
        
        # reference time increment
        if time_increments[0]!=1.0:
            time_increments=[time_increments[i]-time_increments[0]+1.0 for i in range(0,n)]

        # construct time weighted difference matrix
        T = np.zeros((n-2,n))
        for i in range(0,n):
            for j in range(0,n-2):
                if i==j:
                    T[j,i]=1/time_increments[i]
                elif i==j+1:
                    T[j,i]=-1/time_increments[i]
        print(T)
        # construct difference matrix of order k
        i=1
        while i<self.k:
            T=T.dot(T.T)
            i+=1
    
        return T

