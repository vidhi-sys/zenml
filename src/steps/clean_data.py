from zenml import step
import pandas as pd
from typing import Tuple
from src.data_cleaning import (
    DataCleaning,
    DataPreProcessStrategy,
    DataDivideStrategy
)

@step
def clean_data(
    data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    # Step 1: Preprocess data
    preprocess = DataCleaning(data=data, strategy=DataPreProcessStrategy())
    processed_data = preprocess.handle_data()

    # Step 2: Split data
    divide = DataCleaning(data=processed_data, strategy=DataDivideStrategy())
    x_train, x_test, y_train, y_test = divide.handle_data()

    return x_train, x_test, y_train, y_test