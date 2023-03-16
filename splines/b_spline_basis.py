import numpy as np


class B_Spline_Basis:
    """Generates the B-Spline basis functions for a given set of knots and observations

    Note this is a 3-dimensional basis function, i.e. the basis functions are evaluated at each observation
    """

    def __init__(self, x, gamma, order=3):
        self.x = x  # observations
        self.k = len(gamma)  # number of knots
        self.gamma = np.concatenate(
            [np.repeat(gamma[0], order), gamma, np.repeat(gamma[-1], order)]
        )  # knots of length 2m+k
        self.m = order + 1  # order of the B-Spline basis functions

    def B(self, x: np.ndarray):
        """Generates the B-Spline basis functions"""

        # Generate the B-Spline basis functions (knots x basis functions x observations)
        B = np.zeros((self.k + 2 * self.m - 1, self.m, len(x)))

        for j in range(0, self.m):

            for i in range(0, len(self.gamma) - j - 1):

                if j == 0:

                    B[i, j] = np.where((x >= self.gamma[i]) & (x < self.gamma[i + 1]), 1, 0)

                else:
                    lhs, rhs = np.zeros(len(x)), np.zeros(len(x))

                    # for non-duplicate knots, evaluate the basis functions
                    if self.gamma[i] != self.gamma[i + j]:

                        lhs = (x - self.gamma[i]) * B[i, j - 1] / (self.gamma[i + j] - self.gamma[i])

                    if self.gamma[i + j + 1] != self.gamma[i + 1]:
                        rhs = (
                            (self.gamma[i + j + 1] - x) * B[i + 1, j - 1] / (self.gamma[i + j + 1] - self.gamma[i + 1])
                        )

                    B[i, j] = lhs + rhs

        return B[: -self.m + 1, -1, :].T  # return the mth order basis functions
