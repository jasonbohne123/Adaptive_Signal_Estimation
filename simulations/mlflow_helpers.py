from typing import Dict, List

import mlflow


def create_mlflow_experiment(experiment_name, bulk=False):
    """Creates New MLFlow Experiment"""

    exp = mlflow.get_experiment_by_name(name=experiment_name)

    if exp:
        if not bulk:
            print("Experiment already exists")
        experiment_id = exp.experiment_id

    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    # Start MLFlow Run
    mlflow_run = mlflow.start_run(experiment_id=experiment_id)

    # Get MLFlow Run ID
    mlflow_tag = mlflow_run.info.run_id

    return experiment_id, mlflow_run, mlflow_tag


def log_mlflow_params(
    mlflow_run,
    params: Dict[str, str],
    metrics: Dict[str, float],
    tags: List[Dict[str, any]],
    artifact_list: List[any],
    artifact_path: str = None,
):
    """Logs Parameters to MLFlow"""

    # Log Parameters to MLFlow
    mlflow.log_params(params)

    # Log Metrics to MLFlow
    mlflow.log_metrics(metrics)

    # set tags to mlflow
    for tag in tags:
        mlflow.set_tags(tag)

    # Log Artifacts to MLFlow
    for artifact in artifact_list:
        mlflow.log_artifact(artifact)

    # End MLFlow Run
    mlflow.end_run()

    return mlflow_run
