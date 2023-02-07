import matplotlib.pyplot as plt
from tf_constants import get_model_constants, get_simulation_constants

from matrix_algorithms.difference_matrix import Difference_Matrix
from matrix_algorithms.time_difference_matrix import Time_Difference_Matrix
from prior_models.prior_model import Prior
from simulations.mlflow_helpers import create_mlflow_experiment, log_mlflow_params


def prep_signal(sample, true_sol, t=None):
    """Generates and preps our signal"""

    n, k = get_model_constants().get("n"), get_model_constants().get("k")
    if n is None:
        n = len(sample)

    sample = sample[:n].reshape(-1, 1)
    true_sol = true_sol[:n].reshape(-1, 1)

    D = Difference_Matrix(n, k)

    if t is not None:
        t = t[:n]
        D = Time_Difference_Matrix(D, t)

    return sample, true_sol, D


def write_to_files(sample, true_sol, sol, prior_model, true_knots, knots, plot):
    """Write artifacts to mlflow"""

    adaptive_penalty = isinstance(prior_model, Prior)

    # plot to visualize estimation
    if plot:
        plt.figure(figsize=(14, 12))
        plt.plot(true_sol, color="black", label="True Signal", lw=10)
        plt.plot(sample, color="blue", label="Noisy Sample", lw=0.5)
        plt.plot(sol, color="red", label="Reconstructed Estimate", lw=5)
        plt.legend()
        plt.title("Linear Trend Filtering Estimate on Noisy Sample")
        plt.savefig("data/images/tf.png")
        plt.close()

        plt.figure(figsize=(14, 12))
        plt.plot(true_sol, color="black", label="True Signal", lw=10)
        plt.plot(sol, color="red", label="Reconstructed Estimate", lw=5)

        if adaptive_penalty:
            fig, ax = plt.subplots(figsize=(14, 12))
            ax.plot(prior_model.t, prior_model.prior, color="green", label="Original Prior", lw=2.5)  # plots prior
            ax_twin = ax.twinx()
            ax_twin.plot(
                prior_model.t, prior_model.orig_data, color="red", label="Original Data", lw=2.5
            )  # plots original data
            for knot in true_knots:
                ax.axvline(
                    x=prior_model.t[knot], color="black", linestyle="--", lw=2.5, label="True Regime Change"
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
    mse_from_sample,
    mse_from_true,
    expected_prediction_error,
    hausdorff_distance,
    len_true_knots,
    len_reconstructed_knots,
    flags,
):
    """Logs params, metrics, and tags to mlflow"""

    log_mlflow, bulk, include_cv = map(flags.get, ["log_mlflow", "bulk", "include_cv"])

    adaptive_penalty = isinstance(prior_model, Prior)

    # extract params and constants for logging
    cv_folds, cross_validation_size, reference_variance, signal_to_noise = map(
        get_simulation_constants().get, ["cv_folds", "cross_validation_size", "reference_variance", "signal_to_noise"]
    )

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
            adaptive_penalty=adaptive_penalty,
        )
    )

    # create mlflow experiement (if not exists) and run
    experiment_id, run, run_tag = create_mlflow_experiment(exp_name, description=description, bulk=bulk)
    if log_mlflow:

        artifact_list = [
            "data/images/tf.png",
            "data/true_sol.txt",
            "data/noisy_sample.txt",
            "data/sol.txt",
        ]
        if adaptive_penalty:
            artifact_list.append("data/images/prior.png")

        if len_true_knots:
            artifact_list.extend(["data/images/knots.png", "data/knots.txt", "data/true_knots.txt"])

        # Log params, metrics, tags, artifacts
        run_end = log_mlflow_params(
            run,
            params={
                "n": n,
                "k": k,
                "order": order,
                "maxiter": maxiter,
                "maxsliter": maxlsiter,
                "tol": tol,
                "cross_validation": include_cv,
                "no_folds": cv_folds,
                "cross_validation_size": cross_validation_size,
                "adaptive_lambda_p": adaptive_penalty,
                "signal_to_noise": signal_to_noise,
                "reference_variance": reference_variance,
                "k_max": K_max,
            },
            metrics={
                "computation_time": results["computation_time"],
                "optimal_relative_lambda": best_scaler,
                "mse_from_sample": mse_from_sample,
                "mse_from_true": mse_from_true,
                "hausdorff_distance": hausdorff_distance,
                "integrated_squared_prediction_error": expected_prediction_error,
                "len_true_knots": len_true_knots,
                "len_reconstructed_knots": len_reconstructed_knots,
                "knot_difference": len_true_knots - len_reconstructed_knots,
                "gap": results["gap"],
            },
            tags=[{"Adaptive": adaptive_penalty}, {"Cross_Validation": include_cv}, {"Status": results["status"]}],
            artifact_list=artifact_list,
        )
