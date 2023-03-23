import numpy as np

from model_selection.partition import best_fit_polynomial, map_intervals


def generalized_cross_validation(Y, optimal_indices, order, true_knots, verbose=False):
    """Determines optimal number of changepoints based on generalized cross validation"""

    optimal_indices = dict(sorted(optimal_indices.items(), key=lambda x: x[0]))

    # init dictionaries to store the results for optimal k cp
    model_mse = dict.fromkeys(optimal_indices.keys(), 0)
    gcv = dict.fromkeys(optimal_indices.keys(), 0)
    gcv_ratio = dict.fromkeys(optimal_indices.keys(), 0)

    # compute the mse of the true model
    if true_knots is None:
        true_knots = []

    temp_cps = np.unique(np.concatenate([[0], true_knots, [len(Y)]])).astype(int)

    fixed_intervals = map_intervals(Y, temp_cps)

    true_mse = 0
    # compute the sum of squared errors of best fitted polynomial each interval
    for inter in list(fixed_intervals.values()):
        mse = best_fit_polynomial(Y, inter, order=order)

        true_mse += mse

    # compute the mse of the model for each k

    for k_i, cps in optimal_indices.items():

        # pad cps
        temp_cps = np.unique(np.concatenate([[0], cps, [len(Y)]])).astype(int)

        fixed_intervals = map_intervals(Y, temp_cps)

        fixed_mse = 0
        # compute the sum of squared errors of best fitted polynomial each interval

        for inter in list(fixed_intervals.values()):
            mse = best_fit_polynomial(Y, inter, order=order)
            fixed_mse += mse

        # average mse over intervals
        model_mse[k_i] = np.mean(fixed_mse)

        # compute effective number of parameters (arise from trend filtering estimate )
        eff_param = len(cps) + order + 1

        # ratio of gcv scores for subsequent k
        gcv[k_i] = fixed_mse / (len(Y) - eff_param) ** 2

        # compute ratio of gcv scores for subsequent k (initial is 1.0)
        gcv_ratio[k_i] = gcv[k_i] / gcv[k_i - 1] if k_i > 0 else 1.0

    sorted_gcv = sorted(gcv_ratio.items(), key=lambda x: x[1])

    # if optimal cp is first index; no cp are found
    if sorted_gcv[0][0] == 0:
        if verbose:
            print("No changepoints found")

    return sorted_gcv
