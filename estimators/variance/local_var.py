import numpy as np
from scipy.stats import iqr


class LocalVariance:
    """Local Variance Class"""

    def __init__(self, x, y, bandwidth_style, preselected_bandwidth=None):
        self.y = y
        self.x = x
        self.bandwidth_style = bandwidth_style
        self.optimal_bandwidth = preselected_bandwidth if preselected_bandwidth else None

    def fit(self):
        """
        Fits a local volatility estimator on a prior series
        """

        if len(self.y) != len(self.x):
            print("Mismatched Series")
            return None

        local_variance = np.zeros((len(self.x)))

        if self.optimal_bandwidth is not None:
            pass
        else:
            # optimal bandwidth selection are for gaussian kernels
            if self.bandwidth_style == 0:
                bw = 0.9 * min(np.std(self.x), iqr(self.x) / 1.35) / (len(self.x) ** 0.2)
            else:
                bw = 1.06 * np.std(self.x) / (len(self.x) ** 0.2)

            self.optimal_bandwidth = bw

        for i in range(0, len(self.x)):

            candidates = []
            for j in range(0, len(self.x)):
                if abs(self.x[i] - self.x[j]) <= self.optimal_bandwidth:
                    candidates.append(self.y[j])
            local_variance[i] = np.std(candidates)

        return local_variance
