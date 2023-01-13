def get_simulation_constants():
    """Default Simulation Constants for Trend Filtering"""
    constants = {
        "k_points": 25,
        "underlying_dist": "normal",
        "sample_variance": 10e-2,
        "n_sims": 2,
        "shift": 100,
        "cv_folds": 40,
        "cross_validation_size": 0.8,
    }
    return constants


def get_model_constants():
    """Default Model Constants for Trned Filtering"""
    hyperparams = {
        "k": 2,
        "n": 500,
        "gamma": 0.5,
        "alpha": 0.01,
        "beta": 0.5,
        "mu": 2,
        "mu_inc": 1e-10,
        "maxiter": 50,
        "maxlsiter": 50,
        "tol": 1e-5,
    }
    return hyperparams
