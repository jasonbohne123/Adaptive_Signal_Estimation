import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split


def cross_validate_bw(
    kernel_series, penalty_series, approach, seed=1234, train_size=0.75
):

    """
    kernel_series: Series we wish to apply kernel smoothing on
    penalty_series: Series we wish to compute adaptive penalty from
    approach: method for cv
    """

    results = {}
    grid = np.linspace(0.1, 100, 10)  # grid of bw params for kernel

    if approach == 1:
        quantile = np.linspace(0.5, 0.95, 10)  # quantile ranges for severity of penalty
        pairs = np.array([[{i: j} for i in grid] for j in quantile]).flatten()
    else:
        pairs = grid

    old_index = kernel_series.index
    k_seconds = (kernel_series.index - kernel_series.index[0]).total_seconds()
    kernel_series = pd.Series(
        kernel_series.values, index=k_seconds
    )  # convert to total seconds elapsed
    p_seconds = (penalty_series.index - penalty_series.index[0]).total_seconds()
    penalty_series = pd.Series(
        penalty_series.values, index=p_seconds
    )  # convert to total seconds elapsed

    scaled_k = np.max(kernel_series)

    Xk_train, Xk_test, yk_train, yk_test = train_test_split(
        kernel_series.index,
        kernel_series.values,
        train_size=train_size,
        random_state=seed,
    )  # 90/10 split series to kernel smooth
    Xk = pd.Series(yk_train, index=Xk_train).sort_index()
    Yk = pd.Series(yk_test, index=Xk_test).sort_index()

    Xp_train, Xp_test, yp_train, yp_test = train_test_split(
        penalty_series.index,
        penalty_series.values,
        train_size=train_size,
        random_state=seed,
    )  # 90/10 split series to kernel smooth
    Xp = pd.Series(yp_train, index=Xp_train).sort_index()
    Yp = pd.Series(yp_test, index=Xp_test).sort_index()

    if approach == 1:
        for k in pairs:
            bw = list(k.keys())[0]
            quantile = list(k.values())[0]

            smooth = smooth_series(Xk, Xp, scaled_k, bw, quantile, approach, cv=False)
            f = interp1d(
                Xk.index, smooth.values, kind="linear", fill_value="extrapolate"
            )  # assume lienar fit between smoothed observations
            pred = np.sum((f(Yk.index) - Yk.values) ** 2)
            results[(bw, quantile)] = pred

        opt_param = sorted(results.items(), key=lambda x: x[1])[0]
        print(
            f"Optimal CV Bandwidth: {opt_param[0][0]} and Quantile {opt_param[0][1]}: Error {opt_param[1]}"
        )
        cv_smooth = smooth_series(
            kernel_series,
            penalty_series,
            scaled_k,
            opt_param[0][0],
            opt_param[0][1],
            approach,
            cv=True,
        )

    else:
        for k in pairs:

            bw = k
            smooth = smooth_series(Xk, Xp, scaled_k, bw, approach=approach, cv=False)
            f = interp1d(
                Xk.index, smooth.values, kind="linear", fill_value="extrapolate"
            )  # assume lienar fit between smoothed observations
            pred = np.sum((f(Yk.index) - Yk.values) ** 2)
            results[k] = pred

        opt_param = sorted(results.items(), key=lambda x: x[1])[0][0]
        print(f"Optimal CV Bandwidth: {bw}")
        cv_smooth = smooth_series(
            kernel_series, penalty_series, scaled_k, opt_param, approach, cv=True
        )

    return pd.Series(cv_smooth.values, index=old_index)


def smooth_series(
    k_series, p_series, scaled_k, bandwidth, quantile=None, approach=1, cv=True
):

    """

    Kernel Smooths a series under the condition to satisfy an exogenous variable (adaptive penalty)

    In this case the space of viable solutions are not unique so we need some condition to choose a viable one

    Approach 1
        Scales to qth quantile of max_norm (assuming a uniform distribution)

    Approach 2
        Applies Kernel directly on the Adaptive Penalties


    """

    thresh = 1000  # threshold to ignore very small penalty params
    smooth = []
    pen = np.pad(max_norm(p_series.values, adaptive=True, verbose=False), (1, 1))

    for x_i_index, x_i in k_series.items():
        smoothed_val = kernel_smooth(x_i_index, k_series, pen, bandwidth, approach)
        smooth.append(smoothed_val)

    smooth = pd.Series(smooth, index=k_series.index)

    # if approach 1 and we have finished cv; scale to q-th quantile of the penalty value observed at max
    if approach == 1 and cv == True:
        index = np.where(pen > thresh)[0]
        scaled_p = 1 / np.max(np.divide(smooth.values[index], pen[index]))
        # scale by smallest ratio to ensure less than penalty up to threshold
        smooth = smooth * scaled_p * quantile

    # otherwise keep unadjusted to evaluate against test set
    else:
        pass

    return smooth


def compute_kernel(x_0, x_i, bandwidth):
    """
    Given two points x_0 and x_i; compute the kernel
    """
    scale = abs((x_0 - x_i) / bandwidth)  # absolute distance in time

    if scale < 1:

        weight = 0.75 * ((1 - scale) ** 2)

    else:
        weight = 0

    return weight


def kernel_smooth(x0, series, pen, bandwidth, approach=1):
    """
    x0: index of recent observation
    series: observations
    pen: max lambda to scale kernel to
    bandwidth: desired bandwidth param
    Approach: Style of KS under Constraints

    """
    num = 0
    ker = 0

    counter = 0
    for index, val in series.items():

        kernel = compute_kernel(
            x0, index, bandwidth
        )  # compute weight as a function of time
        ker += kernel
        if approach == 1:
            num += kernel * val  # kernel smoothed series constrained by penalty
        else:
            num += kernel * pen[counter]  # kernel smoothed penalty computed by series
        counter += 1

    return num / ker


def max_norm(y, k=2, adaptive=False, verbose=True):

    """

    Computes the smallest max penalty which provides a lower bound of linear affine fits

    * More efficient ways to compute matrix inverses
    * Condition number as proxy for ill-conditioned matrices
    * Reasoning for near zero values at endpoints


    """
    D = Dmat(len(y), k)
    D_DT = np.linalg.inv(np.matmul(D, D.T))

    if not adaptive:
        max_lam = np.max(abs(np.matmul(np.matmul(D_DT, D), y)))
    else:
        max_lam = abs(np.matmul(np.matmul(D_DT, D), y))

    if verbose:
        print("Condition Number is ", round(np.linalg.cond(np.matmul(D, D.T)), 2))
    return max_lam
