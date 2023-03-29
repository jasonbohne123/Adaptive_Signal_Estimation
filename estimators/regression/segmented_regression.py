import sys

sys.path.append("../estimators")
import numpy as np
from base_estimator import Base_Estimator


class Univariate_Segmented_Regression(Base_Estimator):
    """Class for Segmented Regression in Univariate Settings; Similarities to Multiple Structural Change Detection
    References: Partition Regression by Guthery 1974
    """

    def __init__(self, x, y, knots, degree):

        self.x = x
        self.y = y

        # segmented regression specific params
        self.interior_knots = knots
        self.degree = degree
        self.knots = np.concatenate([[0], self.interior_knots, [len(self.y)]])
        self.intervals = self.map_intervals(self.knots)

        # fit the model
        self.coefficients, self.y_hat = self.fit()

    def fit(self, warm_start=False):
        """Fits a polynomial of order k to the data Y across a given interval"""

        # partitions x and y of interval
        coefficients = {}
        estimate = {}

        for idx in self.intervals:

            interval = self.intervals[idx]
            y = self.y[interval[0] : interval[1]]

            x_range = np.arange(interval[0], interval[1], 1)

            poly_coef = np.polyfit(x_range, y, self.degree, full=True)[0]

            coefficients[idx] = poly_coef

            polynomial = poly_coef.T.dot(np.vstack([x_range**i for i in range(self.degree, -1, -1)]))

            estimate[idx] = polynomial

        return coefficients, estimate

    def estimate(self, t: np.ndarray):
        """Predicts the value of y for a given x"""

        t = np.sort(t)

        all_estimates = []

        # split prediction points into intervals and predict accordingly
        for ct, idx in enumerate(self.intervals):
            interval = self.intervals[idx]

            if ct == 0:
                credible_points = np.where(t <= interval[0])

            elif ct == len(self.intervals) - 1:
                credible_points = np.where(t >= interval[1])

            else:

                credible_points = np.where((t >= interval[0]) & (t <= interval[1]))

            estimates = self.coefficients[idx].T.dot(
                np.vstack([t[credible_points] ** i for i in range(self.degree, -1, -1)])
            )

            all_estimates.append(estimates)

        return np.concatenate(all_estimates)

    def map_intervals(self, indices):
        """Maps the indices of the candidate changepoints to a dictionary of intervals"""
        intervals = {}
        for i in range(len(indices) - 1):
            intervals[i] = [indices[i], indices[i + 1]]

        return intervals
