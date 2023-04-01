from estimators.trend_filtering.trend_filter import Trend_Filter


def fit_submodel(x, y, configs, name):
    """Fits a submodel given a name and hyperparameters"""

    # extract relevant hyperparameters given model name
    if name == "Trend_Filter":
        k = configs["k"]
        method = configs["method"]
        submodel = Trend_Filter(x, y, k, method)

    elif name == "Regression_Splines":
        pass
    elif name == "Smoothing_Splines":
        pass
    elif name == "KernelSmoother":
        pass
    elif name == "Univariate_Segmented_Regression":
        pass

    else:
        raise Exception("Submodel not implemented")

    return submodel
