import numpy as np

from model_selection.partition import best_fit_polynomial, map_intervals


def generalized_cross_validation(Y, optimal_indices, order, true_knots, verbose=False):
    """Determines optimal number of changepoints based on generalized cross validation"""

    # init dictionaries to store the results for optimal k cp
    model_mse = dict.fromkeys(optimal_indices.keys(), 0)
    gcv = dict.fromkeys(optimal_indices.keys(), 0)

    # compute the mse of the true model
    temp_cps = np.unique(np.concatenate([[0], true_knots, [len(Y)]])).astype(int)

    fixed_intervals = map_intervals(Y, temp_cps)

    true_mse = 0
    print(fixed_intervals)
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
        model_mse[k_i] = fixed_mse

        # compute effective number of parameters (arise from trend filtering estimate )
        eff_param = len(cps) + order + 1

        print("eff_param", eff_param, "fixed_mse", fixed_mse, "true_mse", true_mse)
        gcv[k_i] = fixed_mse / (len(Y) - eff_param) ** 2

    sorted_gcv = sorted(gcv.items(), key=lambda x: x[1])

    # if optimal cp is first index; no cp are found
    if sorted_gcv[0][0] == 0:
        if verbose:
            print("No changepoints found")

    return sorted_gcv
