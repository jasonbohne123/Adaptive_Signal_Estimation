def get_simulation_constants():
    """Default Simulation Constants for Trend Filtering"""
    constants = {
        "n_sims": 25,
        # simulation params
        "k_points": 4,
        "underlying_dist": "normal",
        "label_style": "k_maxima",
        "signal_to_noise": 0.05,
        "reference_variance": 10e-2,
        "shift": 100,
        # cross validation params
        "cv_folds": 20,
        "cross_validation_size": 0.75,
        "cv_grid_lb": 10e-8,
        "cv_iterations": 5,
        "verbose_cv": True,
    }
    return constants


def get_model_constants():
    """Default Model Constants for Trend Filtering"""
    hyperparams = {
        # Optimization params
        "k": 2,
        "n": 1000,
        "gamma": 0.5,
        "alpha": 0.01,
        "beta": 0.5,
        "mu": 2,
        "mu_inc": 1e-10,
        "maxiter": 100,
        "maxlsiter": 50,
        "tol": 1e-4,
        # model params
        "solve_cp": True,
        "K_max": 8,
        "order": 1,
        "cp_threshold": 0.75 * 10e-2,
    }
    return hyperparams
