import numpy as np

from basis.continous_tf import Continous_TF
from estimators.base_estimator import Base_Estimator
from estimators.regression.helpers.extract_changepoints import extract_cp
from estimators.regression.helpers.partition import partition_solver
from estimators.regression.helpers.segmentation_constants import get_segmentation_constants
from estimators.trend_filtering.helpers.difference_matrix import Difference_Matrix
from model_selection.changepoint_model_selection import ratio_model_selection


class Piecewise_Polynomial_Model(Base_Estimator):
    """Piecewise Polynomial Model with fixed knots selected during fitting"""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        k: int,
        method: str,
    ):
        self.x = x
        self.y = y
        self.k = k
        self.name = "Piecewise_Polynomial_Model"
        self.method = method

        # multiple methods exist for continous time extension
        assert self.method in ["continous_tf", "piecewise_polynomial"]

        self.D = Difference_Matrix(len(self.x), self.k, t=self.x)

        # constants for candidate changepoint selection
        self.quantile = get_segmentation_constants()["cp_quantile"]
        self.K_max = max(int(get_segmentation_constants()["K_max"] * self.x.shape[0]), 1)
        self.nu = get_segmentation_constants()["nu"]
        self.order = get_segmentation_constants()["order"]

        # select optimal knots
        self.knots, self.all_models, self.optimal_model = self.fit()

    def fit(self, warm_start=False):
        """Determine the most significant knots across a candidate range; returning optimal along with all models"""

        reshaped_y = self.y.reshape(1, -1)[0]

        # Extract all candidate knots up to a threshold btwn diff consecutive knots
        candidate_knots = extract_cp(reshaped_y, self.D, self.quantile)

        print(candidate_knots)

        # adjust K_max to the number of candidate knots
        self.K_max = min(self.K_max, len(candidate_knots))

        # Apply dynamic programming to find optimal knots
        dp_set = partition_solver(reshaped_y, candidate_knots, K_max=self.K_max, k=self.order)

        print(dp_set)

        # If no knots are selected, return None
        if dp_set is None:
            return []

        all_models, optimal_model = ratio_model_selection(reshaped_y, dp_set, self.order, self.nu, verbose=True)

        # Get the optimal knots
        knots = dp_set[optimal_model]

        # flag to indicate that knots have been selected
        self.select_knots = True

        return knots, all_models, optimal_model

    def estimate(self, t: np.ndarray):

        if self.method == "continous_tf":

            return Continous_TF(self.y, self.D, self.k).evaluate_tf(t)
        elif self.method == "piecewise_polynomial":
            pass
        else:
            raise ValueError("method not supported")
