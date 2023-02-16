import numpy as np
from difference_matrix import Difference_Matrix
from k_differences import differences


class Time_Difference_Matrix(Difference_Matrix):
    """Time Weighted Difference Matrix for L1 Trend Filtering

    Inherits from Difference_Matrix so original matrices are accessible

    """

    def __init__(self, matrix: Difference_Matrix, t=None):
        super().__init__(matrix.n, matrix.k, matrix.style)
        self.t = t
        self.time_enabled = True

        self.D = matrix

        # if time increments are not provided, assume they are unit spaced
        if t is None:
            t = np.arange(1, self.D.n + 1)

        # returns the time matrix T
        self.T_D = self.construct_time_matrix(t)
        self.T_DDT = self.T_D.dot(self.T_D.T)

        # Note there are issues with LAPACK so will specify sparse for time weighted
        self.T_DDT_inv = np.asarray(self.invert(self.T_DDT, style="sparse"), order="C")

        assert self.T_DDT.dot(self.T_DDT_inv).all() == np.eye(self.D.n - self.D.k).all()

    def construct_time_matrix(self, t):
        """Accounts for unequal time increments via recursion by Tibshirani"""

        D_k = Difference_Matrix(self.D.n, 1).D

        for k in range(1, self.D.k):
            D_1 = Difference_Matrix(self.D.n - k, 1).D

            diff = np.array(differences(t, k=k))
            scale = np.diag((k) / diff)

            D_k = D_1.dot(scale.dot(D_k))

        return D_k
