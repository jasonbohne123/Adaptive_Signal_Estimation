import numpy as np
from difference_matrix import Difference_Matrix


class Time_Difference_Matrix(Difference_Matrix):
    """Time Weighted Difference Matrix for L1 Trend Filtering"""

    def __init__(self, matrix: Difference_Matrix, time_increments):
        super().__init__(matrix.n, matrix.k, matrix.style)
        self.time_increments = time_increments

        self.time_difference_matrix = self.construct_time_matrix()

    def construct_time_matrix(self):
        pass


def adjust_penalty_time(lambda_p, times, k, verbose):
    """Adjusts penalty by difference in time between observations"""

    if times is None:
        if verbose:
            print("No time information provided, using default penalty")
        return lambda_p

    if k != 2:
        if verbose:
            print("Time information provided, but k is not 2, using default penalty")
        return lambda_p

    n = len(times)
    t_diff = np.diff(times)

    # forms three tridiagonal of time penalties in linear trend filtering
    t_diff1 = np.fmax(0.01, np.pad(t_diff, (0, 1), "constant", constant_values=t_diff[-1]))
    t_diff3 = np.fmax(0.01, np.pad(t_diff, (0, 1), "constant", constant_values=t_diff[-1]))
    t_diff2 = t_diff1 + t_diff3

    a = t_diff1 / t_diff3
    b = -t_diff2 / (t_diff1 * t_diff3)
    c = t_diff3 / t_diff1

    # constructs a banded matrix of time differences
    T = dia_matrix((np.vstack([a, b, c]), [0, 1, 2]), shape=(n - 2, n - 2)).toarray()

    # Scales our penalty by the time differences
    lambda_p = np.dot(lambda_p.T, abs(T)).reshape(-1, 1)

    return lambda_p
