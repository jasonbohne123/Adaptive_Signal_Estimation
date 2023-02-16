import math

import numpy as np
from falling_factorial_gen import Falling_Factorial_Basis

from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.k_differences import differences


class Continous_TF:
    def __init__(self, x_tf: np.ndarray, D: Difference_Matrix, k: int):
        self.x_tf = x_tf
        self.k = k

        self.falling_factorial_basis = Falling_Factorial_Basis(self.x_tf, self.k)

        self.h_j_x = self.falling_factorial_basis.h_j_x
        self.h_k_j_x = self.falling_factorial_basis.h_k_j_x

        self.phi = self.compute_phi(self.x_tf)
        self.theta = self.compute_theta(self.x_tf)

    def compute_phi(self, x):
        """Compute phi coefficents for h_j_x terms"""
        self.x_tf[0]

        for j in range(1, self.k + 1):
            1 / math.factorial(j - 1)

            diff = np.diag(1 / np.array(differences(x, k=j)))

    def compute_theta(self, x):
        """Compute theta coefficents for h_k_j_x terms"""

    def evaluate_tf(self, x: np.ndarray):
        """Evaluate the TF at range x"""

        h_j_x = self.h_j_x(x)
        h_k_j_x = self.h_k_j_x(x, self.k)

        return np.dot(self.phi, h_j_x) + np.dot(self.theta, h_k_j_x)
