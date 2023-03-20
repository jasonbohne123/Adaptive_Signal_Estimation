import sys
from collections import Counter

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
        assert (estimators[0].knots == sorted(estimators[0].knots)).all(), "Knots must be sorted"
        assert (estimators[1].knots == sorted(estimators[1].knots)).all(), "Knots must be sorted"

        self.x = estimators[0].knots
        self.y = estimators[1].knots

        # order of splines
        self.k = estimators[0].order
        self.l = estimators[1].order

        self.x_diff = {k_: np.array(differences(self.x, k_)) for k_ in range(1, self.k + 1)}
        self.y_diff = {k_: np.array(differences(self.y, k_)) for k_ in range(1, self.l + 1)}

        self.p = self.k + self.l - 1

        sorted_knots = sorted(np.concatenate([self.x, self.y]))
        self.z = self.construct_z(sorted_knots)
        self.unique_z = np.unique(self.z)

        print(f"Knots for first estimator: {len(self.x)}")
        print(f"Knots for second estimator: {len(self.y)}")
        print(f"Number of knots in byproduct: {len(self.unique_z)}")

        self.e = estimators[0].gamma
        self.f = estimators[1].gamma

        self.len_e = len(self.e)
        self.len_f = len(self.f)

        print(f"Number of parameters for first estimator: {self.len_e}")
        print(f"Number of parameters for second estimator: {self.len_f}")

        h = [i for i in range(len(self.unique_z))]

        # vector of coefficients for the basis functions evaluated at multiple knots
        self.G_h = self.G(h, self.k, self.l)

        self.integral = None

    def construct_z(self, sorted_knots: np.ndarray):

        """Constructs multiplicity of knots in the biproduct"""

        freq_a = Counter(self.x)
        freq_b = Counter(self.y)

        # need to correctly account for multiplicity of knots
        multiplicty = dict(
            zip(
                sorted_knots,
                [
                    max(
                        self.k + freq_a[knot] - 1 if knot in freq_a else self.k,
                        self.l + freq_b[knot] - 1 if knot in freq_b else self.l,
                    )
                    for knot in sorted_knots
                ],
            )
        )

        new_knots = [[knot for i in range(multiplicty[knot])] for knot in multiplicty.keys()]
        new_knots = [knot for sublist in new_knots for knot in sublist]

        return new_knots

    def construct_tau(self, h: list, k, l):

        """Constructs the biproduct of regression coefficients"""

        # (s1-params, s2-params, k, l)
        tau = np.zeros((len(self.e), len(self.f), self.k, self.l))

        # generate byproduct for base case
        def base_case():

            """Base case for the recursion"""

            tau_i_j = np.zeros((self.len_e, self.len_f))

            i = np.where(self.x >= self.x[h])[0][0] if h < len(self.x) else self.len_e - 1
            j = np.where(self.y >= self.y[h])[0][0] if h < len(self.y) else self.len_f - 1

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

                inner_scaler_lhs = 1  # k_ / self.x_diff[k_]

                # above will be nonnegative as always more params than knots for continuous splines
                padded_x = np.pad(self.x, (0, self.len_e - len(self.x)), "constant", constant_values=(0, 0))
                padded_y = np.pad(self.y, (0, self.len_f - len(self.y)), "constant", constant_values=(0, 0))

                term1 = tau[:, :, k_ - 1, l_].T.dot((z_h_p_1 - padded_x))
                term2 = tau[:, :, k_ - 1, l_].T.dot((np.roll(padded_x, k_) - z_h_p_1))

                inner_scaler_rhs = 1  # (l_ / self.y_diff[l_])

                term3 = tau[:, :, k_, l_ - 1].dot(z_h_p_1 - padded_y)
                term4 = tau[:, :, k_, l_ - 1].dot(np.roll(padded_y, l_) - z_h_p_1)

                lhs = (inner_scaler_lhs * (term1 + term2)).reshape(-1, 1)
                rhs = (inner_scaler_rhs * (term3 + term4)).reshape(1, -1)

                tau[:, :, k_, l_] = scaler * (lhs.dot(rhs)).T

        # return a matrix of i,j indices
        return tau[:, :, k - 1, l - 1]

    def G(self, h: np.ndarray, k, l):
        """Given an index set h return G(h)"""

        G_h = 0
        for h_ in h:
            h_ = int(h_)
            # cosntruct the order-indexed polynomials for given knot set h
            tau_i_j = self.construct_tau(h_, k, l)
            e_h = self.e[h_] if h_ < len(self.e) else 1
            f_h = self.f[h_] if h_ < len(self.f) else 1
            G_h += tau_i_j * e_h * f_h

        return G_h
