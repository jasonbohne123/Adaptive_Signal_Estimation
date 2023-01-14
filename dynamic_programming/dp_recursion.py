import numpy as np


def map_intervals(Y, indices):
    """Maps the indices of the candidate changepoints to a dictionary of intervals"""
    intervals = {}
    for i in range(len(indices) - 1):
        intervals[i] = [indices[i], indices[i + 1]]

    return intervals


def best_fit_polynomial(Y, interval, order=1):
    """Fits a polynomial of order k to the data Y across a given interval"""
    # partitions x and y of interval
    y = Y[interval[0] : interval[1]]
    x_range = np.arange(interval[0], interval[1], 1)

    # fit polynomial of order k to data
    poly_coef = np.polyfit(x_range, y, order, full=True)[0]

    polynomial = poly_coef.T.dot(np.vstack([x_range**i for i in range(order, -1, -1)]))

    # mean squared error of fit
    mse = np.mean((y - polynomial) ** 2)

    return mse


def compute_J(Y, indices, cp_mappings, order):
    """Evaluate cost matrix between each nested pair of changepoints"""

    k = len(indices) - 1
    mapping_indices = np.arange(k)

    # initialize cost matrix
    cost_matrix = np.zeros((k, k))

    # evaluate cost matrix, looping over nested pairs of merged intervals
    for i in range(len(mapping_indices)):
        for j in range(i, len(mapping_indices)):

            interval = [cp_mappings[i][0], cp_mappings[j][1]]
            cost_matrix[i, j] = best_fit_polynomial(Y, interval, order=order)

    return cost_matrix


def compute_V(cost_matrix, K_max, indices):
    """Computes the optimal cost and changepoint locations"""

    k = len(indices) - 1

    # initialize optimal cost and optimal location matrices
    optimal_cost = np.zeros((K_max + 1, k))
    optimal_loc = np.zeros((K_max, k))

    # no changepoints; best RSE is the sum of squared errors on interval
    for i in range(k):
        optimal_cost[0, i] = cost_matrix[0, i]

    # loop over number of possible changepoints calculating optimal cost and location of indices
    # utilizes dynamic programming structure to build solution from subproblems solution
    for k_i in range(1, K_max + 1):
        for j in range(k_i + 1, k + 1):

            optimal_cost[k_i, j - 1] = np.min(optimal_cost[k_i - 1, k_i - 1 : j - 1] + cost_matrix[k_i:j, j - 1])
            ind = np.argmin(optimal_cost[k_i - 1, k_i - 1 : j - 1] + cost_matrix[k_i:j, j - 1])
            optimal_loc[k_i - 1, j - 1] = ind + k_i - 1

    return optimal_loc


def optimal_segmentation(optimal_loc, indices, K_max):
    """Computes the optimal segmentation based on recursive RSE"""

    all_loc = optimal_loc.copy()
    k = len(indices) - 1

    # dictionary keyed by potential cp, values are cps that are optimal for that k_i
    total_loc = {}

    for k_i in range(1, K_max + 1):
        total_loc[k_i - 1] = np.zeros(k_i)
        total_loc[k_i - 1][k_i - 1] = all_loc[k_i - 1, k - 1]
        for i in range(k_i - 1, 0, -1):
            total_loc[k_i - 1][i - 1] = all_loc[i - 1, int(total_loc[k_i - 1][i])]

    return total_loc


def convert_observed_cp(optimal_segment, indices):
    """Converts the optimal changepoints to the observed indices"""
    all_segments = optimal_segment.copy()

    # keys by number of points
    for i in range(len(all_segments)):
        indices_dict = dict(zip([i for i in range(0, len(indices))], indices))

        all_segments[i] = [indices_dict[int(i) + 1] for i in all_segments[i]]
    all_segments = {i + 1: all_segments[i] for i in range(len(all_segments))}
    # include case where zero changepoints are optimal
    all_segments[0] = [0]
    return all_segments


def dp_solver(Y, indices, K_max, k, verbose=False):
    """DP Management function to determine optimal changepoints per fixed size"""

    if K_max > len(indices):
        if verbose:
            print("K_max must be less than or equal to the number of candidate changepoints")
        K_max = len(indices)

    indices = np.unique(np.concatenate([[0], indices, [len(Y)]]))
    if verbose:
        print("Candidate Indices are {}".format(indices))

    # map the indices to the intervals of the data
    cp_mappings = map_intervals(Y, indices)

    # Initialize cost matrix
    cost_matrix = compute_J(Y, indices, cp_mappings, k)

    # Compute optimal cost and changepoint locations
    optimal_loc = compute_V(cost_matrix, K_max, indices)

    # Compute optimal segmentation
    optimal_segment = optimal_segmentation(optimal_loc, indices, K_max)

    optimal_indices = convert_observed_cp(optimal_segment, indices)

    return optimal_indices
