def get_simulation_constants():
    constants = {
        "k_points": 10,
        "underlying_dist": "normal",
        "sample_variance": 10e-2,
        "n_sims": 10,
        "shift": 100,
        "cv_folds": 40,
    }
    return constants


def get_model_constants():
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
        "tol": 1e-8,
    }
    return hyperparams
