### Cross Validation Constants
# k_folds : Number of Folds for Cross Validation
# cv_grid_lb : Lower Bound for Grid Search
# grid_size : Number of Grid Points
# grid_spacing : Spacing of Grid Points
# warm_start : Whether to Warm Start Cross Validation
# verbose : Whether to Print Cross Validation Progress


def get_cv_constants():
    return {
        "k_folds": 3,
        "cv_grid_lb": 1e-4,
        "grid_size": 15,
        "grid_spacing": "log",
        "warm_start": False,
        "verbose": True,
    }
