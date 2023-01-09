from typing import Dict, List

import mlflow


def create_mlflow_experiment(experiment_name):
    """Creates New MLFlow Experiment"""

    mlflow.set_tracking_uri(uri="../simulations/mlflow/mlruns")

    exp = mlflow.get_experiment_by_name(name=experiment_name)

    if exp:
        print("Experiment already exists")
        experiment_id = exp.experiment_id

    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    # Start MLFlow Run
    mlflow_run = mlflow.start_run(experiment_id=experiment_id)

    # Get MLFlow Run ID
    mlflow_tag = mlflow_run.info.run_id

    return mlflow_run, mlflow_tag


def log_mlflow_params(mlflow_run: str, params: Dict[str, str], artifact_list: List[any], artifact_path: str = None):
    """Logs Parameters to MLFlow"""

    # Log Parameters to MLFlow
    mlflow.log_params(params)

    for artifact in artifact_list:
        mlflow.log_artifact(artifact, artifact_path)

    # End MLFlow Run
    mlflow.end_run()

    return mlflow_run
