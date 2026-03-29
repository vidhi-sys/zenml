import click as ck
from pipelines.deployment_pipeline import (
    deployment_pipeline,
    inference_pipeline
)

# Modes
DEPLOY = "deploy"
DEPLOY_AND_PREDICT = "deploy_and_predict"
PREDICT = "predict"


@ck.command()
@ck.option(
    "--config",
    "-c",
    type=ck.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT
)
@ck.option(
    "--min-accuracy",
    default=0.92,
)
@ck.option(
    "--data-path",
    default="C:/Users/VIDHI/OneDrive/Desktop/MLFLOW/Data/olist_customers_dataset.csv",
)
def run_deployment(config: str, min_accuracy: float, data_path: str):

    if config == DEPLOY:
        deployment_pipeline(
            data_path=data_path,
            min_accuracy=min_accuracy
        )

    elif config == PREDICT:
        inference_pipeline(
            pipeline_name="deployment_pipeline",
            data_path=data_path
        )

    elif config == DEPLOY_AND_PREDICT:
        deployment_pipeline(
            data_path=data_path,
            min_accuracy=min_accuracy
        )

        inference_pipeline(
            pipeline_name="deployment_pipeline",
            data_path=data_path
        )


