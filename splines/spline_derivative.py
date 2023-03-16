import numpy as np
from regression_spline_estimator import Regression_Spline_Estimator

from matrix_algorithms.k_differences import differences


def compute_spline_derivative(spline: Regression_Spline_Estimator, order: int):

    """Computes the kth derivative of a spline returning a new spline object"""

    assert order > 0, "order must be greater than 0"

    # recursively compute the kth derivative of a spline
    if order > 1:
        return compute_spline_derivative(compute_spline_derivative(spline, order - 1), 1)

    # original spline knots and repeated knots
    knots = spline.knots
    repeated_knots = spline.basis.gamma
    x = spline.x

    # extract the B-Spline basis functions, (N x (K+M))
    basis = spline.basis

    # determine the p-1 order derivative basis function; (N x (K+M+1)
    derivative_basis = basis.B(x, m=spline.order)

    # (K+M)
    gamma = spline.gamma

    # differences between original control points (K+M-1)
    p_diff = gamma[1:] - gamma[:-1]

    # p-differences between original snipped knots (K+2M-2)
    # (K+2M-2-3)
    knot_diff = np.array(differences(repeated_knots[1:-1], spline.order)).reshape(-1, 1)
    index = np.where(knot_diff != 0)[0]

    # (K+2M-2-3) x 1
    Q = np.zeros((derivative_basis.shape[1], 1))

    # update indices with non-zero differences
    Q[index + 1] = spline.order * p_diff[index] * (1 / knot_diff[index])

    # compute the new coefficients
    estimate = derivative_basis.dot(Q)

    regression_spline = Regression_Spline_Estimator(x, estimate, knots, order=spline.order - 1)

    assert len(regression_spline.basis.gamma) == len(repeated_knots) - 2, "derivate of spline has 2 less boundary knots"

    return regression_spline
