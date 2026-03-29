from zenml import step
import pandas as pd
import logging

class IngestData:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_step(data_path: str) -> pd.DataFrame:
    obj = IngestData(data_path)
    return obj.get_data()