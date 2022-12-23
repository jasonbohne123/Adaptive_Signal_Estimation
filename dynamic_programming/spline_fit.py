import numpy as np
from scipy.interpolate import LSQUnivariateSpline


def fit_linear_spline(y, optimal_cp, order=1):
    """Fits a linear spline to the data using the optimal changepoints"""

    x = np.arange(0, len(y), 1)

    spline = LSQUnivariateSpline(x, y, t=optimal_cp, k=order)

    return spline(x)
