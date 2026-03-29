from zenml import step
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from src.evaluation import MSE
import mlflow
from zenml.client import Client
experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None)

def evaluate_model(
    model: RandomForestClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series
) -> float:

    try:
        logging.info("Evaluating model...")

        predictions = model.predict(x_test)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("mse",mse)

        logging.info(f"Model MSE: {mse}")

        return mse

    except Exception as e:
        logging.error(f"Error in evaluation: {e}")
        raise e