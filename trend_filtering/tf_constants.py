def get_simulation_constants():
    """Default Simulation Constants for Trend Filtering"""
    constants = {
        "k_points": 10,
        "underlying_dist": "normal",
        "label_style": "k_maxima",
        "true_variance": 10e-3,
        "sample_variance": 10e-2,
        "n_sims": 2,
        "shift": 100,
        "cv_folds": 25,
        "cross_validation_size": 0.90,
        "cv_grid_lb": 10e-2,
    }
    return constants


def get_model_constants():
    """Default Model Constants for Trend Filtering"""
    hyperparams = {
        "k": 2,
        "n": 500,
        "gamma": 0.5,
        "alpha": 0.01,
        "beta": 0.5,
        "mu": 2,
        "mu_inc": 1e-10,
        "maxiter": 100,
        "maxlsiter": 50,
        "tol": 1e-4,
        "solve_cp": False,
        "K_max": 5,
        "order": 1,
        "cp_threshold": 0.75 * 10e-2,
    }
    return hyperparams
