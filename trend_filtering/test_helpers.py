import matplotlib.pyplot as plt
import numpy as np

from estimators.trend_filtering.helpers.tf_constants import get_model_constants, get_simulation_constants
from matrix_algorithms.difference_matrix import Difference_Matrix
from prior_models.prior_model import Prior
from simulations.mlflow_helpers import create_mlflow_experiment, log_mlflow_params


def prep_signal(sample, true_sol, prior_model=None, t=None):
    """Generates and preps our signal"""

    n, k = get_model_constants().get("n"), get_model_constants().get("k")
    if n is None:
        n = len(sample)

    sample = sample[:n].reshape(-1, 1)
    true_sol = true_sol[:n].reshape(-1, 1)

    if prior_model is not None:
        assert len(prior_model.prior) == len(true_sol)

        prior_model = 1 / prior_model.prior

    if t is not None:
        assert len(t) == len(true_sol)

    D = Difference_Matrix(n, k, prior=prior_model, t=t)
    return sample, true_sol, D


def write_to_files(sample, true_sol, sol, spline, prior_model, true_knots, knots, plot):
    """Write artifacts to mlflow"""

    adaptive_penalty = isinstance(prior_model, Prior)

    if not adaptive_penalty:
        t = np.arange(0, len(true_sol))
    else:
        t = prior_model.t

    # plot to visualize estimation
    if plot:
        plt.figure(figsize=(14, 12))
        plt.plot(t, true_sol, color="black", label="True Signal", lw=10)
        plt.plot(t, sample, color="blue", label="Noisy Sample", lw=0.5)
        plt.plot(t, sol, color="red", label="Reconstructed Estimate", lw=5)
        plt.plot(t, spline, color="green", label="Spline Estimate", lw=3)
        plt.legend()
        plt.title("Linear Trend Filtering Estimate on Noisy Sample")
        plt.savefig("data/images/tf.png")
        plt.close()

        plt.figure(figsize=(14, 12))
        plt.plot(t, true_sol, color="black", label="True Signal", lw=10)
        plt.plot(t, sol, color="red", label="Reconstructed Estimate", lw=5)

        if adaptive_penalty:
            fig, ax = plt.subplots(figsize=(14, 12))
            ax.plot(t, prior_model.prior, color="green", label="Original Prior", lw=2.5)  # plots prior
            ax_twin = ax.twinx()
            ax_twin.plot(t, prior_model.orig_data, color="red", label="Original Data", lw=2.5)  # plots original data
            for knot in true_knots:
                ax.axvline(
                    x=t[knot], color="black", linestyle="--", lw=2.5, label="True Regime Change"
                )  # plots associated cp
            plt.legend()
            plt.title("Original Prior")
            plt.savefig("data/images/prior.png")
            plt.close()

        # vertical lines for regime changes
        if knots:
            for knot in knots:
                plt.axvline(x=knot, color="purple", linestyle="--", lw=2.5, label="Estimated Regime Change")

            for knot in true_knots:
                plt.axvline(x=knot, color="black", linestyle="--", lw=2.5, label="True Regime Change")

        plt.title("Estimated Regime Changes")
        plt.xlabel("Time")
        plt.ylabel("Observation")
        plt.savefig("data/images/knots.png")
        plt.close()

    # save files (eventually refactor custom model)
    with open("data/true_sol.txt", "w") as f:
        f.write(str(true_sol))

    with open("data/noisy_sample.txt", "w") as f:
        f.write(str(sample))

    with open("data/sol.txt", "w") as f:
        f.write(str(sol))

    if knots:
        with open("data/knots.txt", "w") as f:
            f.write(str(knots))

        with open("data/true_knots.txt", "w") as f:
            f.write(str(true_knots))

    if isinstance(prior_model, Prior):
        with open("data/prior.txt", "w") as f:
            f.write(str(prior_model.orig_data))


def log_to_mlflow(
    exp_name,
    results,
    prior_model,
    best_scaler,
    adaptive_results,
    non_adaptive_results,
    snr,
    len_true_knots,
    len_reconstructed_knots,
    flags,
):
    """Logs params, metrics, and tags to mlflow"""

    log_mlflow, bulk, include_cv, time_aware, adaptive_tf = map(
        flags.get, ["log_mlflow", "bulk", "include_cv", "time_aware", "adaptive_tf"]
    )

    mse_from_sample, mse_from_true, spline_mse, hausdorff_distance, expected_prediction_error = map(
        adaptive_results.get,
        ["mse_from_sample", "mse_from_true", "spline_mse", "hausdorff_distance", "expected_prediction_error"],
    )

    # comparison to non adaptive metrics (if applicable)
    if non_adaptive_results:
        (non_adapt_mse_from_true, non_adapt_spline_mse, non_adapt_hausdorff_distance,) = map(
            non_adaptive_results.get,
            ["mse_from_true", "spline_mse", "hausdorff_distance"],
        )

        mse_true_diff = non_adapt_mse_from_true - mse_from_true
        spline_mse_diff = non_adapt_spline_mse - spline_mse
        hausdorff_distance_diff = non_adapt_hausdorff_distance - hausdorff_distance

    # extract params and constants for logging
    cv_folds, cross_validation_size, reference_variance, signal_to_noise = map(
        get_simulation_constants().get, ["cv_folds", "cross_validation_size", "reference_variance", "signal_to_noise"]
    )
    signal_to_noise = snr if snr else signal_to_noise

    k, n, maxiter, maxlsiter, tol, K_max, order = map(
        get_model_constants().get, ["k", "n", "maxiter", "maxlsiter", "tol", "K_max", "order"]
    )

    description = (
        "Linear Trend Filtering on Noisy Sample with Cross Validation of {cv_folds} folds and "
        "Cross Validation Size of {cross_validation_size}  Reference Variance of {reference_variance} "
        " Signal to Noise Ratio of {signal_to_noise} and Adaptive Penalty of {adaptive_penalty}".format(
            cv_folds=cv_folds,
            cross_validation_size=cross_validation_size,
            reference_variance=reference_variance,
            signal_to_noise=signal_to_noise,
            adaptive_penalty=adaptive_tf,
        )
    )

    # create mlflow experiement (if not exists) and run
    experiment_id, run, run_tag = create_mlflow_experiment(exp_name, description=description, bulk=bulk)
    if log_mlflow:

        params = {
            "n": n,
            "k": k,
            "order": order,
            "maxiter": maxiter,
            "maxsliter": maxlsiter,
            "tol": tol,
            "cross_validation": include_cv,
            "no_folds": cv_folds,
            "cross_validation_size": cross_validation_size,
            "adaptive_lambda_p": adaptive_tf,
            "signal_to_noise": signal_to_noise,
            "reference_variance": reference_variance,
            "k_max": K_max,
        }

        metrics = {
            "computation_time": results["computation_time"],
            "optimal_relative_lambda": best_scaler,
            "mse_from_sample": mse_from_sample,
            "mse_from_true": mse_from_true,
            "spline_mse": spline_mse,
            "hausdorff_distance": hausdorff_distance,
            "integrated_squared_prediction_error": expected_prediction_error,
            "len_true_knots": len_true_knots,
            "len_reconstructed_knots": len_reconstructed_knots,
            "knot_difference": len_true_knots - len_reconstructed_knots,
            "gap": results["gap"],
        }

        artifact_list = [
            "data/images/tf.png",
            "data/true_sol.txt",
            "data/noisy_sample.txt",
            "data/sol.txt",
        ]

        # add prior and knots if applicable
        if adaptive_tf:
            artifact_list.append("data/images/prior.png")
            metrics["bandwidth"] = prior_model.bandwidth if prior_model.name == "Kernel_Smooth_Prior" else None
            if non_adaptive_results:
                metrics.update(
                    {
                        "mse_true_diff": mse_true_diff,
                        "spline_mse_diff": spline_mse_diff,
                        "hausdorff_distance_diff": hausdorff_distance_diff,
                    }
                )

        if len_true_knots:
            artifact_list.extend(["data/images/knots.png", "data/knots.txt", "data/true_knots.txt"])

        # Log params, metrics, tags, artifacts
        run_end = log_mlflow_params(
            run,
            params=params,
            metrics=metrics,
            tags=[
                {"Adaptive": adaptive_tf},
                {"Cross_Validation": include_cv},
                {"Status": results["status"]},
                {"Time_Aware": time_aware},
            ],
            artifact_list=artifact_list,
        )
