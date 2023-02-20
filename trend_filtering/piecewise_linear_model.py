import sys

import numpy as np
from scipy.interpolate import LSQUnivariateSpline

sys.path.append("../")

from matrix_algorithms.difference_matrix import Difference_Matrix
from model_selection.cp_model_selection import generalized_cross_validation
from model_selection.partition import partition_solver
from trend_filtering.continous_tf import Continous_TF
from trend_filtering.helpers import extract_cp
from trend_filtering.tf_constants import get_model_constants


class Piecewise_Linear_Model:
    """Piecewise Linear Model which is callable for prediction from optimization"""

    def __init__(
        self,
        x: np.ndarray,
        D: Difference_Matrix,
        select_knots=False,
        true_knots=None,
    ):
        self.x = x

        self.k = D.k

        self.D = D
        self.t = D.t

        # constants for cp selection and model
        self.threshold = get_model_constants()["cp_threshold"]
        self.K_max = get_model_constants()["K_max"]
        self.order = get_model_constants()["order"]
        self.select_knots = select_knots
        self.true_knots = true_knots

        if self.select_knots:
            self.knots = self.get_knots()

        # extract tf to continous domain

        self.continous_tf = Continous_TF(self.x, self.D, self.k)

    def predict(self, t: np.ndarray):
        """Predict the output at time t using continous extrapolation"""

        predict = self.continous_tf.evaluate_tf(t)

        return predict

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
