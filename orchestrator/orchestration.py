from pathlib import Path

import mlflow

from prefect import flow, task
from prefect import get_run_logger
from prefect.task_runners import SequentialTaskRunner

from google.cloud import storage

from dotenv import load_dotenv  # TODO change credentials to secrets manager

load_dotenv("orchestrator/.env")
client = storage.Client()

# TODO move to separate file
tracking_uri = "https://mlflow-mgwuu2wdea-uc.a.run.app"
git_url = "https://github.com/caldasdeoliveira/mlops-zoomcamp-final-project-model.git"
experiment_name = "NYC_taxi_experiment"
version = "dev"


@task
def source_dataset(
    start_date: str = "2020-10",
    number_of_months: int = 1,
):

    params = {
        "start_date": start_date,
        "number_of_months": number_of_months,
    }

    run = mlflow.projects.run(
        git_url,
        parameters=params,
        entry_point="source_dataset",
        experiment_name=experiment_name,
        version=version,
    )
    return run.run_id


@task
def prep_data(filename: str):  # TODO correct path typehint
    params = {
        "filename": filename,
    }

    run = mlflow.projects.run(
        git_url,
        parameters=params,
        entry_point="prep_data",
        experiment_name=experiment_name,
        version=version,
    )
    return run.run_id


@task
def train(X_dataset: str, y_dataset: str):  # TODO correct path typehint
    params = {
        "X_dataset": X_dataset,
        "y_dataset": y_dataset,
    }

    run = mlflow.projects.run(
        git_url,
        parameters=params,
        entry_point="train",
        experiment_name=experiment_name,
        version=version,
    )
    return run.run_id


@flow  # (task_runner=SequentialTaskRunner())
def train_flow(log_stdout=True):
    logger = get_run_logger()
    mlflow.set_tracking_uri(tracking_uri)

    logger.info("sourcing dataset")
    source_dataset_run = source_dataset()
    logger.info("finished sourcing dataset")
    artifact_root_sourcing = mlflow.get_run(source_dataset_run).info.artifact_uri

    bucket = artifact_root_sourcing.split("/")[2]
    prefix = Path(*artifact_root_sourcing.split("/")[3:])

    dataset_list = [
        f"gs://{bucket}/{b.name}"
        for b in client.list_blobs(
            bucket,
            prefix=prefix,
        )
        if b.name.endswith(".parquet")
    ]
    logger.info(dataset_list)

    assert dataset_list, "Sourced dataset list is empty."

    prep_data_run = prep_data(dataset_list[0])
    artifact_root_prep = mlflow.get_run(prep_data_run).info.artifact_uri

    bucket = artifact_root_prep.split("/")[2]
    prefix = Path(*artifact_root_prep.split("/")[3:])

    dataset_list = [
        f"gs://{bucket}/{b.name}"
        for b in client.list_blobs(
            bucket,
            prefix=prefix,
        )
        if b.name.endswith(".parquet")
    ]
    logger.info(dataset_list)

    X_dataset = [d for d in dataset_list if d.startswith("X_")]
    assert X_dataset, "X dataset is empty."
    X_dataset = X_dataset[0]

    y_dataset = [d for d in dataset_list if d.startswith("y_")]
    assert y_dataset, "y dataset is empty."
    y_dataset = y_dataset[0]

    train_run = train(X_dataset, y_dataset)


if __name__ == "__main__":
    train_flow()
