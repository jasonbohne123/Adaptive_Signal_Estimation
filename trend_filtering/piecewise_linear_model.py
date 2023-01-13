from typing import Union

import numpy as np

from dynamic_programming.cp_model_selection import generalized_cross_validation
from dynamic_programming.dp_recursion import dp_solver
from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from trend_filtering.helpers import extract_cp
from trend_filtering.tf_constants import get_model_constants


class Piecewise_Linear_Model:
    """Piecewise Linear Model which is callable for prediction from optimization"""

    def __init__(
        self,
        x: np.ndarray,
        D: Union[Difference_Matrix, Time_Difference_Matrix, None] = None,
        t: np.ndarray = None,
        k=2,
        threshold=10e-4,
        select_knots=False,
    ):
        self.x = x

        if not isinstance(D, Difference_Matrix) and not isinstance(D, Time_Difference_Matrix):

            self.D = Difference_Matrix(len(self.x), k)

            if t is not None:
                self.T = Time_Difference_Matrix(self.D, t)
                self.time_enabled = True

        self.k = k
        self.K_max = get_model_constants()["K_max"]
        self.order = get_model_constants()["order"]
        self.threshold = threshold
        self.select_knots = select_knots

        if self.select_knots:
            self.knots = self.get_knots()

    def predict(self, t: np.ndarray):
        """Predict the output at time t"""

        rhs_val = max(len(self.x), max(t) + 1)

        # if precomputed knots are used, use them
        if self.select_knots:
            knots = self.knots

        # else use naive values
        else:
            knots = np.arange(rhs_val)

        t = list(t)
        estimate = []

        for t_i in t:

            # index of farthest right left point of t_i
            left_knots = knots[np.where(knots < t_i)[0]]

            # index of farthest left right point of t_i
            right_knots = knots[np.where(knots > t_i)[0]]

            if len(left_knots) == 0 and len(right_knots) > 0:
                left_knot = 0
                right_knot = right_knots[0]

            elif len(right_knots) == 0 and len(left_knots) > 0:
                left_knot = left_knots[-1]
                right_knot = rhs_val - 1
            elif len(left_knots) == 0 and len(right_knots) == 0:
                left_knot = 0
                right_knot = rhs_val - 1
            else:
                left_knot = left_knots[-1]
                right_knot = right_knots[0]

            # left point will always be observed (even if extrapolation)
            left_point = self.x[min(left_knot, len(self.x) - 1)]

            # right point sometimes is extrapolation
            right_point = (
                self.x[right_knot]
                if right_knot < len(self.x)
                else (self.x[-1] - self.x[-2]) * (right_knot - left_knot) + left_point
            )

            slope = (right_point - left_point) / (right_knot - left_knot)

            # prediction for t_i is linear between left and right point
            estimate.append(left_point + slope * (t_i - left_knot))

        return np.array(estimate)

    def get_knots(self):
        """Get the knots of the piecewise linear model up to a threshold"""

        if self.time_enabled:
            D = self.T.T_D
        else:
            D = self.D

        # Extract all candidate knots up to a threshold
        candidate_knots = extract_cp(self.x, D, self.threshold)

        # Apply dynamic programming to find optimal knots
        dp_set = dp_solver(self.x, candidate_knots, K_max=self.K_max, k=self.order)

        # Select optimal knots via generalized cross validation
        optimal_trend_cp_mse, optimal_trend_cp_gcv = generalized_cross_validation(dp_set, self.x, self.order)

        knots = dp_set[optimal_trend_cp_gcv[0][0]]

        self.select_knots = True

        return knots
