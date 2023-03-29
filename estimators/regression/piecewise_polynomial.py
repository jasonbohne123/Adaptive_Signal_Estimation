import numpy as np

from estimators.base_estimator import Base_Estimator
from estimators.regression.helpers.extract_changepoints import extract_cp
from estimators.regression.helpers.partition import partition_solver
from estimators.regression.helpers.segmentation_constants import get_segmentation_constants
from model_selection.changepoint_model_selection import ratio_model_selection


class Piecewise_Polynomial_Model(Base_Estimator):
    """Piecewise Polynomial Model with fixed knots selected during fitting"""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        k: int,
    ):
        self.x = x
        self.y = y
        self.k = k

        # constants for candidate changepoint selection
        self.quantile = get_segmentation_constants()["cp_quantile"]
        self.K_max = max(int(get_segmentation_constants()["K_max"] * self.x.shape[0]), 1)
        self.nu = get_segmentation_constants()["nu"]
        self.order = get_segmentation_constants()["order"]

        # select optimal knots
        self.knots, self.all_models, self.optimal_model = self.fit()

    def fit(self, warm_start=False):
        """Determine the most significant knots across a candidate range; returning optimal along with all models"""

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

    def estimate(self, t: np.ndarray):

        pass
