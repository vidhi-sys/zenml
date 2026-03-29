import logging
import numpy as np
from sklearn.metrics import mean_squared_error
from abc import ABC, abstractmethod


class Evaluation(ABC):
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass
    # ground truth and model prediction
class MSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            return mse
        
        except Exception as e:
            logging.error("error in mse {}".format(e))
            raise e