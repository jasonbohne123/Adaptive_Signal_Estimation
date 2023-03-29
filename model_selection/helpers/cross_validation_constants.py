### Cross Validation Constants
# k_folds : Number of Folds for Cross Validation
# cv_grid_lb : Lower Bound for Grid Search
# grid_size : Number of Grid Points
# grid_spacing : Spacing of Grid Points


def get_cv_constants():
    return {
        "k_folds": 20,
        "cv_grid_lb": 1e-4,
        "grid_size": 5,
        "grid_spacing": "log",
        "verbose": True,
    }
