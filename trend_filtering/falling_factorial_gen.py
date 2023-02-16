import numpy as np


class Falling_Factorial_Basis:
    def __init__(self, t: np.ndarray, k: int):
        self.t = t.flatten()
        self.k = k
        self.h_j_x, self.h_k_j_x = self.construct_basis()

    def construct_basis(self):

        h_j_x = self.construct_truncated_basis()

        h_k_j_x = self.construct_constraint_basis()

        return (h_j_x, h_k_j_x)

    def construct_truncated_basis(self):
        """Constructs the truncated basis functions callable by x"""

        def h_j_x(x: np.ndarray):
            terms = np.zeros((self.k + 1, len(x)))

            terms[0, :] = 1
            for j in range(1, self.k + 1):
                terms[j, :] = np.prod([(x - self.t[l - 1]) for l in range(1, j)], axis=0)

            return terms

        return h_j_x

    def construct_constraint_basis(self):
        """Constructs the constraint basis functions callable by x"""

        def h_k_j_x(x: np.ndarray, k: int = self.k):
            n = len(self.t)

            terms = np.zeros((n - k - 1, len(x)))

            for j in range(1, n - k):

                terms[j - 1, :] = np.prod(
                    [(x - self.t[j + l - 2]) * (x >= self.t[j + k - 2]).astype(int) for l in range(1, k + 1)], axis=0
                )

            return terms

        return h_k_j_x
