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
        "mu_inc": 1e-10,
        "maxiter": 25,
        "maxlsiter": 25,
        "tol": 1e-6,
    }
    return hyperparams
