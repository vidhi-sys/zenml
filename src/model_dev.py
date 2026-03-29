import logging
from sklearn.ensemble import RandomForestClassifier
from abc import ABC, abstractmethod

class model(ABC):
    @abstractmethod
    def train(self, x_train, y_train):
        pass

class RF(model):
    def train(self, x_train, y_train, **kwargs):
        try:
            rforest = RandomForestClassifier(**kwargs)
            rforest.fit(x_train, y_train)
            logging.info("Model trained via Random Forest")
            return rforest

        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise e