import numpy as np

from evaluation_metrics.loss_functions import compute_error
from model_selection.partition import best_fit_polynomial, map_intervals
from trend_filtering.tf_constants import get_simulation_constants


def generalized_cross_validation(Y, optimal_indices, order, true_knots, verbose=False):
    """Determines optimal number of changepoints based on generalized cross validation"""

    # init dictionaries to store the results for optimal k cp
    model_mse = dict.fromkeys(optimal_indices.keys(), 0)
    biased_cv_epe = dict.fromkeys(optimal_indices.keys(), 0)
    gcv = dict.fromkeys(optimal_indices.keys(), 0)

    # compute the mse of the true model
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
        model_mse[k_i] = fixed_mse

    # max deviation between candidate prior and true prior
    # here we use the haussdorf distance
    max_deviation = 0
    max_deviation_dict = {}
    for k_i, cps in optimal_indices.items():
        deviation = compute_error(np.array(true_knots), np.array(cps), type="hausdorff")
        max_deviation_dict[k_i] = deviation

        if deviation > max_deviation:
            max_deviation = deviation
            max_deviation_set = cps

    max_distance = compute_error(np.array(true_knots), np.array(max_deviation_set), type="hausdorff")
    cv_bias = get_simulation_constants()["cv_bias"]

    # compute the biased cross validation error for each k
    for k_i, mse in model_mse.items():

        # perhaps instead of using the max deviation, we use the number of parameters as a measure of simplicity
        current_distance = max_deviation_dict[k_i]

        relative_accuracy = cv_bias * (mse / true_mse)
        in_sample_simplicity = (1 - cv_bias) * (current_distance / max_distance) ** 2

        biased_cv_epe[k_i] = relative_accuracy + in_sample_simplicity

        # compute effective number of parameters;
        # this is order + 1 for each interval (# of knots + 1)

        eff_param = (k_i + 1) * (order + 1)

        gcv[k_i] = mse / (len(Y) - eff_param) ** 2

    sorted_gcv = sorted(gcv.items(), key=lambda x: x[1])

    # if optimal cp is first index; no cp are found
    if sorted_gcv[0][0] == 0:
        if verbose:
            print("No changepoints found")

    return sorted_gcv
