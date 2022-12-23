import numpy as np
from scipy.stats import iqr

# TO:DO incoporate outlier robust bandwidth calibration
# At the moment is senstive to outliers


class KernelSmoother:
    """Kernel Smoother Class"""

    def __init__(self, x, y, bandwidth_style):
        self.y = y
        self.x = x
        self.bandwidth_style = bandwidth_style
        self.optimal_bandwidth = None

    def compute_kernel(self, x_0, x_i, bandwidth):
        """Given two points x_0 and x_i; compute the gaussian kernel utilizing euclidean distance"""
        if bandwidth == 0:
            return 0
        scale = abs((x_0 - x_i) / bandwidth)

        weight = np.exp(-(scale**2))

        return weight

    def fit(self):
        """
        Fits a kernel smoothing estimator on a prior series
        """

        if len(self.y) != len(self.x):
            print("Mismatched Series")
            return None

        kernel_matrix = np.zeros((len(self.x), len(self.x)))

        # optimal bandwidth selection are for gaussian kernels
        if self.bandwidth_style == 0:
            bw = 0.9 * min(np.std(self.x), iqr(self.x) / 1.35) / (len(self.x) ** 0.2)
        else:
            bw = 1.06 * np.std(self.x) / (len(self.x) ** 0.2)

        self.optimal_bandwidth = bw

        for i in range(0, len(self.x)):

            for j in range(0, len(self.x)):
                kernel = self.compute_kernel(self.x[i], self.x[j], bandwidth=self.optimal_bandwidth)
                kernel_matrix[i, j] = kernel

        fitted_kernel_matrix = kernel_matrix / np.sum(kernel_matrix, axis=1)
        return fitted_kernel_matrix

    def smooth_series(self, fitted_kernel_matrix):
        """Smooths the series using the kernel smoothing estimator"""
        smooth_prior = fitted_kernel_matrix.dot(self.y)

        return smooth_prior

    def evaluate_kernel(self, y_i):
        """Evaluates the kernel at a given point y_i"""

        if self.optimal_bandwidth is None:
            if self.bandwidth_style == 0:
                bw = 0.9 * min(np.std(self.x), iqr(self.x) / 1.35) / (len(self.x) ** 0.2)
            else:
                bw = 1.06 * np.std(self.x) / (len(self.x) ** 0.2)
            self.optimal_bandwidth = bw

        kernel_matrix = np.zeros((len(self.x)))
        for i in range(0, len(self.x)):
            kernel = self.compute_kernel(self.x[i], y_i, self.optimal_bandwidth)
            kernel_matrix[i] = kernel

        # decay weight causes extrapolation estimate to be zero
        if np.sum(kernel_matrix) == 0:
            return None

        y_hat = np.sum(kernel_matrix * self.y) / np.sum(kernel_matrix)

        return y_hat
