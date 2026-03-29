import numpy as np
import pandas as pd

from zenml import pipeline, step

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from src.steps.ingest_data import ingest_step
from src.steps.clean_data import clean_data
from src.steps.model_train import train_model
from src.steps.model_evaluate import evaluate_model



@step
def deployment_trigger(
    mse: float,
    max_mse: float,
) -> bool:

    print(f"Model MSE: {mse}")
    print(f"Threshold (max MSE): {max_mse}")

    return mse <= max_mse



@step
def prediction_service_loader(
    pipeline_name: str,
    step_name: str,
    running: bool = True,
) -> MLFlowDeploymentService:

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=step_name,
        running=running,
    )

    if not services:
        raise RuntimeError(
            "No MLflow deployment service found. Run deployment first."
        )

    return services[0]



@step
def predictor(
    service: MLFlowDeploymentService,
    data: pd.DataFrame,
) -> np.ndarray:

    service.start(timeout=10)
    predictions = service.predict(data)

    return predictions



@pipeline(enable_cache=True)
def deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.92,  
):

    df = ingest_step(data_path=data_path)

    x_train, x_test, y_train, y_test = clean_data(df)

    model = train_model(x_train=x_train, y_train=y_train)

   
    mse = evaluate_model(
        model=model,
        x_test=x_test,
        y_test=y_test,
    )

    
    deploy_decision = deployment_trigger(
        mse=mse,
        max_mse=min_accuracy,
    )

  
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy_decision,
    )


@pipeline(enable_cache=False)
def inference_pipeline(
    pipeline_name: str,
    data_path: str,
):

    
    df = ingest_step(data_path=data_path)

   
    _, x_test, _, _ = clean_data(df)

  
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        step_name="mlflow_model_deployer_step",
    )

  
    predictions = predictor(
        service=service,
        data=x_test,
    )