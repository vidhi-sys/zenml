from zenml import step
import pandas as pd
from src.model_dev import RF
import mlflow
from zenml.client import Client
from zenml.client import Client

client = Client()
experiment_tracker = client.active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None)
def train_model(
    x_train: pd.DataFrame,
   # x_test: pd.DataFrame,
    y_train: pd.Series,
    #y_test: pd.Series
):
    mlflow.sklearn.autolog()
    model = RF()
    trained_model = model.train(x_train, y_train)
    return trained_model