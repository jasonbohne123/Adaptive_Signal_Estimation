### Model Hyperparameters
# alpha : Line Search Parameter for Newton's Method (Boyd)
# beta : Line Search Parameter for Newton's Method (Boyd)
# mu : Initial Step Parameter for Newton's Method (Boyd)
# mu_inc : Step Parameter Increment for Newton's Method (Boyd)
# maxiter : Maximum Number of Iterations for Newton's Method (Boyd)
# maxlsiter : Maximum Number of Iterations for Line Search (Boyd)


def get_model_constants():
    """Default Model Constants for Trend Filtering"""
    hyperparams = {
        "k": 1,
        "n": 250,
        "alpha": 0.01,
        "beta": 0.5,
        "mu": 2,
        "mu_inc": 1e-10,
        "maxiter": 40,
        "maxlsiter": 30,
        "tol": 1e-4,
    }

    return hyperparams
