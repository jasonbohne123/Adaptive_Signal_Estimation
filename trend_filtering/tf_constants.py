### Parameter Significance
# Reference Variance: Specifies the variance of the true process
# Signal to Noise: Specifies the signal to noise ratio of the Noisy Sample

# CV Folds: Number of folds for cross validation grid search
# Cross Validation Size: Size of the cross validation sample
# CV Grid LB: Lower bound of the grid for cross validation of regularization parameter
# CV Iterations: Number of iterations for cross validation


def get_simulation_constants():
    """Default Simulation Constants for Trend Filtering"""
    constants = {
        "n_sims": 5,
        "n_samples": 1,
        # simulation params
        "underlying_dist": "normal",
        "label_style": "k_local_spikes",
        "signal_to_noise": 1,  # in time-sensitive case, this should be much larger
        "reference_variance": 1e-2,
        "shift": 100,
        # cross validation params
        "cv_folds": 20,
        "cross_validation_size": 0.75,
        "cv_grid_lb": 1e-4,
        "cv_iterations": 5,
        "cv_bias": 0.5,
        "verbose_cv": True,
    }
    return constants


### Model Hyperparameters
# alpha : Line Search Parameter for Newton's Method (Boyd)
# beta : Line Search Parameter for Newton's Method (Boyd)
# mu : Initial Step Parameter for Newton's Method (Boyd)
# mu_inc : Step Parameter Increment for Newton's Method (Boyd)
# maxiter : Maximum Number of Iterations for Newton's Method (Boyd)
# maxlsiter : Maximum Number of Iterations for Line Search (Boyd)

# solve_cp : Whether to solve the changepoint problem
# K_max : Maximum Number of Changepoints
# cp_threshold : Threshold for candidate changepoints


def get_model_constants():
    """Default Model Constants for Trend Filtering"""
    hyperparams = {
        # Optimization params
        "k": 1,  # linear trend filtering
        "n": 250,
        "alpha": 0.01,
        "beta": 0.5,
        "mu": 2,
        "mu_inc": 1e-10,
        "maxiter": 40,
        "maxlsiter": 30,
        "tol": 1e-6,
        # model params
        "solve_cp": True,
        "K_max": 5,
        "order": 1,
        "cp_quantile": 0.02,  # good proxy is 0.1 of variance of true process
        "min_cp_distance": 25,
    }
    return hyperparams
