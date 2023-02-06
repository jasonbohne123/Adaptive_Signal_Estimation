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
        "n_sims": 10,
        # simulation params
        # "k_points": 3, (only required for k_maxima and k_minima)
        "underlying_dist": "normal",
        "label_style": "k_local_spikes",
        "signal_to_noise": 0.075,
        "reference_variance": 1e-2,
        "shift": 100,
        # cross validation params
        "cv_folds": 10,
        "cross_validation_size": 0.75,
        "cv_grid_lb": 1e-8,
        "cv_iterations": 5,
        "verbose_cv": True,
    }
    return constants


### Model Hyperparameters
# gamma : Adaptive Step Size for Newton's Method
# alpha : Line Search Parameter for Newton's Method
# beta : Line Search Parameter for Newton's Method
# mu : Initial Step Parameter for Newton's Method
# mu_inc : Step Parameter Increment for Newton's Method
# maxiter : Maximum Number of Iterations for Newton's Method
# maxlsiter : Maximum Number of Iterations for Line Search

# solve_cp : Whether to solve the changepoint problem
# K_max : Maximum Number of Changepoints
# cp_threshold : Threshold for candidate changepoints


def get_model_constants():
    """Default Model Constants for Trend Filtering"""
    hyperparams = {
        # Optimization params
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
        # model params
        "solve_cp": True,
        "K_max": 5,
        "order": 1,
        "cp_threshold": 1e-2,
    }
    return hyperparams
