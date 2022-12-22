
def get_hyperparams():
    """
    Returns the hyperparameters for the adaptive trend filtering algorithm

    Returns
    -------
    hyperparams : dict
        Dictionary containing the hyperparameters
    """
    hyperparams = {
        "gamma": 0.5,
        "alpha": 0.01,
        "beta": 0.5,
        "mu": 2,
        "maxiter": 25,
        "maxlsiter": 25,
        "tol": 1e-4,
    }
    return hyperparams