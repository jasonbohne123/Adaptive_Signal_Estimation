import mlflow
import numpy as np

from simulations.Conditional_Simulator import ConditionalSimulator


def generate_conditional_piecewise_paths(
    prior, sim_style, label_style="k_maxima", k_points=10, underling_dist="normal", n_sims=10000, evaluate=False
):
    """Generate piecewise constant/linear paths with changepoints at the k_maxima of the prior distribution"""

    sim = ConditionalSimulator(prior, sim_style, label_style, k_points, underling_dist, n_sims=n_sims)
    true = sim.simulate()

    if not evaluate:
        return true
    else:
        return sim.evaluate(sim.cp_index, true)


def apply_function_to_paths(paths, function):
    """Apply a function to each path in a set of simulations"""

    # Ideally we can improve this from naivealy applying function once at a time

    # Thoughts

    return np.apply_along_axis(function, 1, paths)


def save_to_mlflow(paths, mlflow_run, mlflow_tag):
    """Save paths to mlflow"""

    ## DECIDE
    # Log each path
    # Plot Results; estimation vs True
    # Log each simulation
    # Plot Sample Results
    # Plot Overall statistics

    mlflow.log_artifact(paths, mlflow_run.info.run_id, mlflow_tag)
