from typing import Union

import numpy as np

from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix


class Piecewise_Linear_Model:
    """Piecewise Linear Model which is callable for prediction from optimization"""

    def __init__(
        self,
        x: np.ndarray,
        D: Union[Difference_Matrix, Time_Difference_Matrix, None] = None,
        t: np.ndarray = None,
        k=2,
        threshold=10e-4,
        precomputed=False,
    ):
        self.x = x

        if not isinstance(D, Difference_Matrix) and not isinstance(D, Time_Difference_Matrix):

            self.D = Difference_Matrix(len(self.x), k)

            if t is not None:
                self.T = Time_Difference_Matrix(self.D, t)
                self.time_enabled = True

        self.k = k
        self.threshold = threshold
        self.precomputed = precomputed

        if self.precomputed:
            self.knots = self.get_knots()

    def predict(self, t: np.ndarray):
        """Predict the output at time t"""

        # precompute knots first predict call
        if not self.precomputed:
            self.knots = self.get_knots()
            self.precomputed = True
            print("Precomputed knots: {}".format(self.knots))

        t = list(t)
        estimate = []

        for t_i in t:

            left_knots = self.knots[np.where(self.knots < t_i)[0]]
            right_knots = self.knots[np.where(self.knots > t_i)[0]]

            if len(left_knots) == 0 and len(right_knots) > 0:
                left_knot = 0
                right_knot = self.knots[0]
            elif len(right_knots) == 0 and len(left_knots) > 0:
                left_knot = self.knots[-1]
                right_knot = len(self.x) - 1
            elif len(left_knots) == 0 and len(right_knots) == 0:
                left_knot = 0
                right_knot = len(self.x) - 1
            else:
                left_knot = left_knots[-1]
                right_knot = right_knots[0]

            left_point = self.x[left_knot]
            right_point = self.x[right_knot]

            slope = (right_point - left_point) / (right_knot - left_knot)

            estimate.append(left_point + slope * (t_i - left_knot))

        return np.array(estimate)

    def get_knots(self):
        """Get the knots of the piecewise linear model up to a threshold"""

        if self.time_enabled:
            D = self.T.T_D
        else:
            D = self.D

        # get the indices of the knots
        knots = np.where(np.abs(D.dot(self.x)) > self.threshold)[0]

        return knots
