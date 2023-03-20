import sys
from collections import Counter
from typing import list

import numpy as np

sys.path.append("../")
from regression_spline_estimator import Regression_Spline_Estimator

from matrix_algorithms.k_differences import differences


class Biprod_B_Spline_Basis:
    """Generates the B-Spline basis functions for the biproduct of two sets of knots and observations

    Note this is a 4-dimensional basis function, i.e. the basis functions are evaluated at each observation for each pair of observations
    """

    def __init__(self, estimators: list[Regression_Spline_Estimator]):

        assert len(estimators) == 2, "Must have two estimators at the moment"

        # knot sets of splines
        assert estimators[0].knots == sorted(estimators[0].knots), "Knots must be sorted"
        assert estimators[1].knots == sorted(estimators[1].knots), "Knots must be sorted"

        self.x = estimators[0].basis.gamma
        self.y = estimators[1].basis.gamma

        # order of splines
        self.k = estimators[0].order
        self.l = estimators[1].order

        self.x_diff = {k_: differences(self.x, k_) for k_ in range(1, self.k + 1)}
        self.y_diff = {k_: differences(self.y, k_) for k_ in range(1, self.l + 1)}

        self.p = self.k + self.l - 1

        self.z = self.construct_z()

        self.e = estimators[0].gamma
        self.f = estimators[1].gamma

        # construct biproduct of regression coefficients
        self.tau = self.construct_tau()

        # vector of coefficients for the basis functions evaluated at multiple knots
        self.G_h = self.G(self.z, self.k, self.l)

        self.integral = None

    def construct_z(self):

        """Constructs multiplicity of knots in the biproduct"""

        freq_a = Counter(self.x)
        freq_b = Counter(self.y)

        # construct new knot set
        multiplicty = dict(zip(self.x, np.fmax(self.k + freq_a.values() - 1, self.l + freq_b.values() - 1)))

        new_knots = [[i for i in range(multiplicty[knot])] for knot in multiplicty.keys()]
        new_knots = [knot for sublist in new_knots for knot in sublist]

        return new_knots

    def construct_tau(self, h: np.ndarray, k, l):

        """Constructs the biproduct of regression coefficients"""

        tau = np.zeros((len(self.e), len(self.f), self.k, self.l))

        assert [type(h[i]) == int for i in range(len(h))], "h must be a list of integers"

        # generate byproduct for base case
        def base_case():

            """Base case for the recursion"""

            tau_i_j = np.zeros((len(self.e), len(self.f)))

            for h_ in h:
                i = self.x.index(self.x[self.x >= h_][0])
                j = self.y.index(self.y[self.y >= h_][0])
                tau_i_j[i, j] = 1

            return tau_i_j

        # initialize the base case
        tau[:, :, 0, 0] = base_case()

        # recursion
        for k_ in range(1, k):
            for l_ in range(1, l):
                p = k_ + l_ - 1

                z_h = self.z[h]
                z_h_p = self.z[h + p - 1]
                z_h_p_1 = self.z[h + p - 2]

                scaler = (z_h_p - z_h) / (p * z_h_p_1 - z_h)

                inner_scaler_lhs = k_ / self.x_diff[k_]

                term1 = tau[:, :, k_ - 1, l_] * (z_h_p_1 - self.x)
                term2 = tau[:, :, k_ - 1, l_] * (np.roll(self.x, k_) - z_h_p_1)

                inner_scaler_rhs = (l_ / self.y_diff[l_]) * (z_h_p_1 - self.y)

                term3 = tau[:, :, k_, l_ - 1] * (z_h_p_1 - self.y)
                term4 = tau[:, :, k_, l_ - 1] * (np.roll(self.y, l_) - z_h_p_1)

                tau[:, :, k_, l_] = scaler * (inner_scaler_lhs * (term1 + term2) + inner_scaler_rhs * (term3 + term4))

        # return a matrix of i,j indices
        return tau[:, :, k - 1, l - 1]

    def G(self, h: np.ndarray, k, l):
        """Given an index set h return G(h)"""

        # cosntruct the order-indexed polynomials for given knot set h
        tau_i_j = self.construct_tau(h, k, l)

        # construct the scaler coefficient
        G_h = tau_i_j.dot(self.e).dot(self.f.T)

        return G_h

    def B(self, x: np.ndarray, m=1):
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

        return B[: self.k + 2 * self.m - m, m - 1, :].T  # return the mth order basis functions
