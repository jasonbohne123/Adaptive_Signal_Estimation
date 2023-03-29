import sys
from collections import Counter

import numpy as np

sys.path.append("../")
from regression_spline_estimator import Regression_Spline_Estimator

from matrix_algorithms.k_differences import differences

##################################################################
# Notes

# Differentiation and Integration of the B-Spline Basis Functions is relatively straightforward

# The difficulty comes in the convoluation of the basis coefficients which requires a 5-tensor indexed by
# the number of basis functions in byproduct, number of basis functions in each semester and the order of each Regression_Spline_Estimator

# I believe this will compress correctly to a 3-tensor indexed by the number of basis functions in byproduct and the order of each Regression_Spline_Estimator

# Issue right now is getting the matrix addition within the recursion to align. Hunch is that by computing across all h-values requires a reformulation of the RecursionError

# Still also need to update B-Spline Basis to have knots of higher order multiplicity


class Biprod_B_Spline_Basis:
    """Generates the B-Spline basis functions for the biproduct of two sets of knots and observations

    Note this is a 4-dimensional basis function, i.e. the basis functions are evaluated at each observation for each pair of observations
    """

    def __init__(self, estimators: list[Regression_Spline_Estimator]):

        assert len(estimators) == 2, "Must have two estimators at the moment"

        # knot sets of splines
        assert (estimators[0].knots == sorted(estimators[0].knots)).all(), "Knots must be sorted"
        assert (estimators[1].knots == sorted(estimators[1].knots)).all(), "Knots must be sorted"

        # knots of the first estimator
        self.x_knots = estimators[0].basis.gamma
        self.y_knots = estimators[1].basis.gamma

        # order of splines( highest degree + 1)
        self.k = estimators[0].m
        self.l = estimators[1].m

        self.p = self.k + self.l - 1

        # sort the knots (with duplicates)
        sorted_knots = sorted(np.concatenate([self.x_knots, self.y_knots]))
        self.z = self.construct_z(sorted_knots)

        # unique knots
        self.unique_z = np.unique(sorted_knots)

        print(f"Knots with duplicates for first estimator: {len(self.x_knots)}")
        print(f"Knots with duplicates for second estimator: {len(self.y_knots)}")
        print(f"Number of knots with duplicates in byproduct: {len(self.z)}")
        print(f"Number of unique knots in byproduct: {len(self.unique_z)}")

        # precompute the differences for quick access in recursion
        self.x_diff = {k_: np.array(differences(self.x_knots, k_)) for k_ in range(1, self.k + 1)}
        self.y_diff = {l_: np.array(differences(self.y_knots, l_)) for l_ in range(1, self.l + 1)}
        self.z_p_diff = {p_: np.array(differences(self.z, p_)) for p_ in range(1, self.p + 1)}
        self.z_p_1_diff = {p_: np.array(differences(self.z, p_ - 1)) for p_ in range(1, self.p + 1)}

        self.e = estimators[0].gamma
        self.f = estimators[1].gamma

        # pad the parameters with zeros if they are not enough
        self.len_e = max(len(self.e), len(self.x_knots))
        self.len_f = max(len(self.f), len(self.y_knots))

        print(" ")
        print(f"Number of parameters for first estimator: {self.len_e}")
        print(f"Number of parameters for second estimator: {self.len_f}")

        # unique basis functions (will be h=K+2M)
        h = [i for i in range(len(self.z))]

        print(f"Number of basis functions: {len(h)}")

        # vector of coefficients for the basis functions evaluated at multiple knots
        self.G_h = self.G(h, self.k, self.l)

        self.integral = None

    def construct_z(self, sorted_knots: np.ndarray):

        """Constructs multiplicity of knots in the biproduct"""

        freq_a = Counter(self.x_knots)
        freq_b = Counter(self.y_knots)

        # need to correctly account for multiplicity of knots
        multiplicty = {
            knot: max(
                self.k + freq_a[knot] - 1 if knot in freq_a else self.k,
                self.l + freq_b[knot] - 1 if knot in freq_b else self.l,
            )
            for knot in sorted_knots
        }

        new_knots = [[knot for i in range(multiplicty[knot])] for knot in multiplicty.keys()]
        new_knots = [knot for sublist in new_knots for knot in sublist]

        return np.array(new_knots)

    def construct_tau(self, h: list, k, l):

        """Constructs the biproduct of regression coefficients"""

        # (crossproduct, s1-params, s2-params, k, l)
        tau = np.zeros((len(h), self.len_e, self.len_f, self.k, self.l))

        print(f"Shape of tau: {tau.shape}")

        # generate byproduct for base case
        def base_case():

            """Base case for the recursion"""

            tau_i_j = np.zeros((len(h), self.len_e, self.len_f))

            for ct, h_ in enumerate(h):

                # find x/y index greater or equal than current knot
                i = np.where(self.x_knots >= self.z[h_])[0][0]
                j = np.where(self.y_knots >= self.z[h_])[0][0]

                # compute the area of the rectangle
                x = self.x_knots[i + 1] - self.x_knots[i] if i < self.len_e - 1 else 0
                y = self.y_knots[j + 1] - self.y_knots[j] if j < self.len_f - 1 else 0
                z = self.z[h_ + 1] - self.z[h_] if h_ < len(self.z) - 1 else 0

                if z == 0:
                    continue

                tau_i_j[ct, i, j] = x * y / z

            return tau_i_j

        # initialize the base case
        tau[:, :, :, 0, 0] = base_case()

        # recursion for the rest of the cases
        for k_ in range(1, k):
            for l_ in range(1, l):
                p = k_ + l_ - 1

                # fetch boundary knots
                z_h = self.z.take(h)
                z_h_p = self.z.take([h_ + p if h_ + p < len(self.z) else len(self.z) - 1 for h_ in h])

                z_h_p_1 = self.z.take([h_ + p - 1 if h_ + p - 1 < len(self.z) else len(self.z) - 1 for h_ in h])

                scaler = np.zeros(len(z_h_p))
                idx_1 = np.where(p * z_h_p_1 != z_h)
                scaler[idx_1] = (z_h_p[idx_1] - z_h[idx_1]) / (p * z_h_p_1[idx_1] - z_h[idx_1])

                inner_scaler_lhs = np.zeros(len(self.x_knots))
                idx1 = np.where(self.x_diff[k_] != 0)
                inner_scaler_lhs[idx1] = k_ / self.x_diff[k_][idx1]

                # for each x_i apply the z transform
                first_diff = np.array([z_i - self.x_knots for z_i in z_h_p])
                second_diff = np.array([np.roll(self.x_knots, k_) - z_i for z_i in z_h_p_1])

                print(f"First diff: {first_diff.shape}")
                print(f"Second diff: {second_diff.shape}")

                term1 = np.dot(first_diff, tau[0, :, :, k_ - 1, l_])
                term2 = np.dot(second_diff, tau[0, :, :, k_ - 1, l_])

                print(f"Term1: {term1.shape}")
                print(f"Term2: {term2.shape}")

                inner_scaler_rhs = np.zeros(len(self.y_knots))
                idx2 = np.where(self.y_diff[l_] != 0)
                inner_scaler_rhs[idx2] = l_ / self.y_diff[l_][idx2]

                first_diff = np.array([z_i - self.y_knots for z_i in z_h_p])
                second_diff = np.array([np.roll(self.y_knots, l_) - z_i for z_i in z_h_p_1])

                print(f"First diff: {first_diff.shape}")
                print(f"Second diff: {second_diff.shape}")

                term3 = np.dot(first_diff, tau[0, :, :, k_, l_ - 1].T)
                term4 = np.dot(second_diff, tau[0, :, :, k_, l_ - 1].T)

                print(f"Term3: {term3.shape}")
                print(f"Term4: {term4.shape}")

                lhs = np.multiply(np.multiply((term1 + term2), inner_scaler_rhs).T, scaler)
                rhs = np.multiply(np.multiply((term3 + term4), inner_scaler_lhs).T, scaler)

                tau[0, :, :, k_, l_] = np.dot(lhs, rhs.T).T

        # return a matrix of i x j indices (unique given input h)
        return tau[0, :, :, k - 1, l - 1]

    def G(self, h: np.ndarray, k, l):
        """Given an index set h return G(h)"""

        assert (type(h[i]) == int for i in range(len(h))), "h must be a list of integers"

        tau = self.construct_tau(h, k, l)

        print(tau)

        G_h = tau.dot(self.f).dot(self.e)

        return G_h
