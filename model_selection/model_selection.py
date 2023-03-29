import numpy as np

from model_selection.partition import best_fit_polynomial, map_intervals


def ratio_model_selection(Y, optimal_indices, order, true_knots, nu, verbose=True):
    """Determines optimal number of changepoints based on generalized cross validation"""

    optimal_indices = dict(sorted(optimal_indices.items(), key=lambda x: x[0]))

    # init dictionaries to store the results for optimal k cp
    model_mse = dict.fromkeys(optimal_indices.keys(), 0)
    dict.fromkeys(optimal_indices.keys(), 0)
    model_ratio = dict.fromkeys(list(optimal_indices.keys())[:-1], 0)

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

        if verbose:
            print("k: {} mse: {}".format(k_i, fixed_mse))

        # average mse over intervals
        model_mse[k_i] = np.sum(fixed_mse)

        # compute ratio of model mses for subsequent k (initial is 1.0)
        if k_i > 0:
            model_ratio[k_i - 1] = model_mse[k_i] / model_mse[k_i - 1]

    # optimal model is the most parsimonious model that is (1-nu) times better than the previous model
    candidate_models = {k: model_ratio[k] for k in model_ratio.keys() if model_ratio[k] > 1 - nu}

    # if no candidate models are found, return optimal model of no cp
    optimal_model = min(candidate_models.keys()) if len(candidate_models) > 0 else 0

    return model_ratio, optimal_model


### Alternative formulation is to utilize a GCV score; which R. Tib. originally proposed a linear approximation to in the seminal lasso shrinkage paper

# compute effective number of parameters (arise from trend filtering estimate )
# eff_param = len(cps) + order + 1
###   gcv[k_i] = fixed_mse / (len(Y) - eff_param) ** 2
