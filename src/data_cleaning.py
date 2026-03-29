import pandas as pd
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple
from sklearn.model_selection import train_test_split

# Abstract base strategy
class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data: pd.DataFrame):
        pass



class DataPreProcessStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data.fillna({
                "product_weight_g": data["product_weight_g"].median(),
                "product_length_cm": data["product_length_cm"].median(),
                "product_height_cm": data["product_height_cm"].median(),
                "product_width_cm": data["product_width_cm"].median(),
                "review_comment_message": "No review"
            }, inplace=True)

            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
                errors="ignore"
            )

            data = data.select_dtypes(include=[np.number])

            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1, errors="ignore")

            return data

        except Exception as e:
            logging.error(f"Error in preprocessing: {e}")
            raise e



class DataDivideStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Tuple:
        try:
            target_column = "review_score"

            if target_column not in data.columns:
                raise ValueError(f"{target_column} not found in dataset")

            X = data.drop(target_column, axis=1)
            y = data[target_column]

            x_train, x_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            return x_train, x_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error in dividing data: {e}")
            raise e


# Context Class
class DataCleaning:

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self):
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e