def get_trend_filtering_simulation_constants():
    constants = {
        "sample_variance": 10e-3,
        "n_sims": 500,
        "k_points": 5,
        "underlying_dist": "normal",
        "cv_folds": 20,
        "n": 500,
        "k": 2,
    }
    return constants
