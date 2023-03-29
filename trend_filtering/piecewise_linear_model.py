import sys

import numpy as np

sys.path.append("../")

from estimators.trend_filtering.helpers.tf_constants import get_model_constants
from matrix_algorithms.difference_matrix import Difference_Matrix
from model_selection.model_selection import ratio_model_selection
from model_selection.partition import partition_solver
from trend_filtering.continous_tf import Continous_TF
from trend_filtering.helpers import extract_cp


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

        # constants for candidate changepoint selection
        self.quantile = get_model_constants()["cp_quantile"]
        self.K_max = max(int(get_model_constants()["K_max"] * self.x.shape[0]), 1)
        self.nu = get_model_constants()["nu"]
        self.order = get_model_constants()["order"]
        self.select_knots = select_knots
        self.true_knots = true_knots

        if self.select_knots:
            self.knots, self.all_models, self.optimal_model = self.get_knots()

        self.continous_tf = Continous_TF(self.x, self.D, self.k)

    def predict(self, t: np.ndarray):
        """Predict the output at time t using continous extrapolation"""

        predict = self.continous_tf.evaluate_tf(t)

        return predict

    def get_knots(self):
        """Get the knots of the piecewise linear model up to a threshold"""

        reshaped_x = self.x.reshape(1, -1)[0]

        # Extract all candidate knots up to a threshold
        candidate_knots = extract_cp(reshaped_x, self.D, self.quantile)

        # adjust K_max to the number of candidate knots
        self.K_max = min(self.K_max, len(candidate_knots))

        # Apply dynamic programming to find optimal knots
        dp_set = partition_solver(reshaped_x, candidate_knots, K_max=self.K_max, k=self.order)

        # If no knots are selected, return None
        if dp_set is None:
            return []

        all_models, optimal_model = ratio_model_selection(
            reshaped_x, dp_set, self.order, self.true_knots, self.nu, verbose=True
        )

        # Get the optimal knots
        knots = dp_set[optimal_model]

        # flag to indicate that knots have been selected
        self.select_knots = True

        return knots, all_models, optimal_model
