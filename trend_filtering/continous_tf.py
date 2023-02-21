import math
import sys

import numpy as np
from falling_factorial_gen import Falling_Factorial_Basis

sys.path.append("../")
from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.k_differences import differences


class Continous_TF:
    def __init__(self, x_tf: np.ndarray, D: Difference_Matrix, k: int, cv: bool = False):
        self.x_tf = x_tf.flatten()
        self.k = k

        self.D = D

        # requires t to be indexed in CV
        assert self.D.t is not None if cv else True

        # in cv case, t is indexed in D
        if self.D.t is not None:
            self.t = self.D.t

        # in non-cv case, t is just the index
        else:
            self.t = np.arange(1, len(self.x_tf) + 1)

        # requires prior to be indexed in CV
        assert self.D.prior is not None if cv else True

        self.prior = self.D.prior

        # create falling factorial basis on input data
        self.falling_factorial_basis = Falling_Factorial_Basis(self.t, self.k)

        self.h_j_x = self.falling_factorial_basis.h_j_x
        self.h_k_j_x = self.falling_factorial_basis.h_k_j_x

        self.phi = self.compute_phi()
        self.theta = self.compute_theta()

    def compute_phi(self):
        """Compute phi coefficents for h_j_x terms"""

        phi = []
        phi.append(self.x_tf[0])

        n = len(self.t)

        # recursive formula for phi
        for j in range(2, self.k + 2):

            scale = 1 / math.factorial(j - 1)

            diff = np.diag(1 / np.array(differences(self.t, k=j - 1)))

            # requires arbitrary difference matrix ( partial of prior and time)
            D = Difference_Matrix(n, k=j - 2, t=self.t, prior=self.prior)

            # time indexed difference matrix
            D_j_1 = D.D

            first_row = np.dot(diff, D_j_1)[0:1, :]

            phi_to_add = scale * np.dot(first_row, self.x_tf.reshape(-1, 1))

            phi.append(phi_to_add[0][0])

        return np.array(phi).reshape(-1, 1)

    def compute_theta(self):
        """Compute theta coefficents for h_k_j_x terms"""

        # note that the difference matrix is (k+1) order

        theta = self.D.D.dot(self.x_tf.reshape(-1, 1)) / math.factorial(self.k)

        return theta

    def evaluate_tf(self, x: np.ndarray):
        """Evaluate the TF at range x"""

        h_j_x = self.h_j_x(x)

        h_k_j_x = self.h_k_j_x(x, self.k)

        return (np.dot(self.phi.T, h_j_x) + np.dot(self.theta.T, h_k_j_x)).flatten()
