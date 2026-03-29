from zenml import pipeline
from src.steps.ingest_data import ingest_step
from src.steps.clean_data import clean_data
from src.steps.model_train import train_model
from src.steps.model_evaluate import evaluate_model

@pipeline(enable_cache=True)
def training_pipeline(data_path: str):

    df = ingest_step(data_path)

    x_train, x_test, y_train, y_test = clean_data(df)

    model = train_model(x_train, x_test, y_train, y_test)

    mse = evaluate_model(model, x_test, y_test)