import numpy as np


class Univariate_Segmented_Regression:
    """Class for Segmented Regression in Univariate Settings; Similarities to Multiple Structural Change Detection
    References: Partition Regression by Guthery 1974
    """

    def __init__(self, x, y, knots, degree):

        self.x = x
        self.y = y
        self.knots = knots
        self.degree = degree

        self.intervals = self.map_intervals(self.knots)

        self.coefficients, self.estimate = self.fit()

    def map_intervals(self, indices):
        """Maps the indices of the candidate changepoints to a dictionary of intervals"""
        intervals = {}
        for i in range(len(indices) - 1):
            intervals[i] = [indices[i], indices[i + 1]]

        return intervals

    def fit(self):
        """Fits a polynomial of order k to the data Y across a given interval"""

        coefficients = {}
        estimate = []

        for idx in self.intervals:

            interval = self.intervals[idx]
            y = self.y[interval[0] : interval[1]]

            x_range = np.arange(interval[0], interval[1], 1)

            poly_coef = np.polyfit(x_range, y, self.degree, full=True)[0]

            coefficients[idx] = poly_coef

            polynomial = poly_coef.T.dot(np.vstack([x_range**i for i in range(self.degree, -1, -1)]))

            estimate.append(polynomial)

        return coefficients, np.concatenate(estimate)

    def predict(self, x: np.ndarray):
        """Predicts the value of y for a given x"""

        x = np.sort(x)

        all_predictions = []

        # split prediction points into intervals
        for idx in self.intervals:
            interval = self.intervals[idx]

            credible_points = np.where((x >= interval[0]) & (x <= interval[1]))

            predictions = self.coefficients[idx].T.dot(
                np.vstack([x[credible_points] ** i for i in range(self.degree, -1, -1)])
            )

            all_predictions.append(predictions)

        return np.concatenate(all_predictions)
