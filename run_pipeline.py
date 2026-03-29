from pipelines.training_pipeline import training_pipeline
from zenml.client import Client
if __name__ == "__main__":
    # insert get tracking url here to get the tracking and then paste the link here in most bottom to see ui
    pipeline_instance = training_pipeline(
        data_path="C:/Users/VIDHI/OneDrive/Desktop/MLFLOW/Data/olist_customers_dataset.csv"
    )
    