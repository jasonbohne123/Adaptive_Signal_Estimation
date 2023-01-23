import numpy as np

from evaluation_metrics.haussdorf import compute_haussdorf_distance
from trend_filtering.tf_constants import get_simulation_constants


def compute_error(x, x_hat, type="mse"):
    assert type in ["mse", "mae", "epe", "hausdorff"]

    if type == "mse" or type == "mae" or type == "epe":

        assert x.shape == x_hat.shape

    if type == "mse":
        return np.sum(np.abs(x - x_hat) ** 2) / len(x)

    elif type == "mae":
        return np.sum(np.abs(x - x_hat)) / len(x)

    elif type == "epe":
        var_y = get_simulation_constants().get("reference_variance")

        mse = compute_error(x, x_hat, type="mse")

        return var_y + mse

    elif type == "hausdorff":
        return compute_haussdorf_distance(x, x_hat)

    else:
        raise ValueError("Error type not recognized")
