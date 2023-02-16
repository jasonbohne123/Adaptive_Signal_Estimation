import sys
from typing import Union

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

sys.path.append("../")

from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from model_selection.cp_model_selection import generalized_cross_validation
from model_selection.partition import partition_solver
from trend_filtering.helpers import extract_cp
from trend_filtering.tf_constants import get_model_constants


class Piecewise_Linear_Model:
    """Piecewise Linear Model which is callable for prediction from optimization"""

    def __init__(
        self,
        x: np.ndarray,
        D: Union[Difference_Matrix, Time_Difference_Matrix, None] = None,
        t: np.ndarray = None,
        select_knots=False,
        true_knots=None,
    ):
        self.x = x

        self.k = get_model_constants()["k"]

        # if D is not provided, create it either with or without time
        if not isinstance(D, Difference_Matrix) and not isinstance(D, Time_Difference_Matrix):

            self.D = Difference_Matrix(len(self.x), self.k)

            if t is not None:
                self.D = Time_Difference_Matrix(self.D, t)

        # constants for cp selection and model
        self.threshold = get_model_constants()["cp_threshold"]
        self.K_max = get_model_constants()["K_max"]
        self.order = get_model_constants()["order"]
        self.select_knots = select_knots
        self.true_knots = true_knots

        if self.select_knots:
            self.knots = self.get_knots()

    def predict(self, t: np.ndarray):
        """Predict the output at time t using linear itnerpolation between two observed values"""

        rhs_val = max(len(self.x), max(t) + 1)

        # if precomputed knots are used, use them
        if self.select_knots:
            knots = self.knots

        # else construct knots from in-sample data
        else:
            knots = np.setdiff1d(np.arange(rhs_val + 1), t)
            knots = np.sort(knots)

        t = list(t)
        estimate = []

        for t_i in t:

            # index of farthest right left point of t_i
            left_knots = knots[np.where(knots < t_i)[0]]

            # index of farthest left right point of t_i
            right_knots = knots[np.where(knots > t_i)[0]]

            # determine if left or right point is extrapolation
            if len(left_knots) == 0 and len(right_knots) > 0:
                left_knot = min(knots)
                right_knot = right_knots[0]

            elif len(right_knots) == 0 and len(left_knots) > 0:
                left_knot = left_knots[-1]
                right_knot = min(rhs_val - 1, max(knots))
            elif len(left_knots) == 0 and len(right_knots) == 0:
                left_knot = min(knots)
                right_knot = min(rhs_val - 1, max(knots))
            else:
                left_knot = left_knots[-1]
                right_knot = right_knots[0]

            # determine farthest right left state of t_i
            left_point = self.x[np.where(knots == left_knot)[0][0]]

            # determine farthest left right state of t_i (might be extrapolation)
            right_point = (
                self.x[np.where(knots == right_knot)[0][0]]
                if right_knot < max(knots)
                else (self.x[-1] - self.x[-2]) * (right_knot - left_knot) + left_point
            )

            slope = (right_point - left_point) / (right_knot - left_knot) if right_knot != left_knot else 0

            # prediction for t_i is linear between left and right point
            estimate.append(left_point + slope * (t_i - left_knot))

        return np.array(estimate)

    def get_knots(self):
        """Get the knots of the piecewise linear model up to a threshold"""

        reshaped_x = self.x.reshape(1, -1)[0]

        # Extract all candidate knots up to a threshold
        candidate_knots = extract_cp(reshaped_x, self.D, self.threshold)

        # Apply dynamic programming to find optimal knots
        dp_set = partition_solver(reshaped_x, candidate_knots, K_max=self.K_max, k=self.order)

        # If no knots are selected, return None
        if dp_set is None:
            return []

        optimal_trend_cp_gcv = generalized_cross_validation(
            reshaped_x, dp_set, self.order, self.true_knots, verbose=True
        )

        # Get the optimal knots
        knots = dp_set[optimal_trend_cp_gcv[0][0]]

        # flag to indicate that knots have been selected
        self.select_knots = True

        return knots

    def fit_linear_spline(self):
        """Fits a linear spline to the data using the optimal changepoints

        Allows for a Continous Fit of the data

        """

        if not self.select_knots:
            self.knots = self.get_knots()

        t = np.arange(0, len(self.x), 1)

        # fits a linear spline to the data with fixed changepoints and order
        spline = LSQUnivariateSpline(t, self.x, t=self.knots, k=self.order)

        return spline(t).reshape(-1, 1)
