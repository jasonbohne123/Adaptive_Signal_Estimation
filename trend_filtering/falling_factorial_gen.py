import numpy as np


class Falling_Factorial_Basis:
    def __init__(self, x: np.ndarray, k: int):
        self.x = x
        self.k = k
        self.h_j_x, self.h_k_j_x = self.construct_basis()

    def construct_basis(self):

        h_j_x = self.construct_truncated_basis()

        h_k_j_x = self.construct_constraint_basis()

        return (h_j_x, h_k_j_x)

    def construct_truncated_basis(self):
        """Constructs the truncated basis functions callable by x"""

        def h_j_x(x: np.ndarray):
            terms = [[x - self.x[l] for l in range(j)] for j in range(self.k + 1)]
            return np.prod(terms, axis=1)

        return h_j_x

    def construct_constraint_basis(self):
        """Constructs the constraint basis functions callable by x"""

        def h_k_j_x(x: np.ndarray, k: int = self.k):
            n = len(self.x)

            # basis function with max indicator
            terms = [
                [(x - self.x[j + l]) * (x >= self.x[j + k]).astype(int) for l in range(k)] for j in range(0, n - k)
            ]

            return np.prod(terms, axis=1)

        return h_k_j_x
