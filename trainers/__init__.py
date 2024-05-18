from . import *
import numpy as np
import pandas as pd


class TrainerAdapter:

    def __init__(self, model_name: str, data: pd.DataFrame, test_ratio=0.2):
        self.model_name = model_name
        self.data = data.copy()
        self.test_ratio = test_ratio

        self.model = None
        self.train_values: np.array = None
        self.test_values: np.array = None
        self.predictions: np.array = None
        self.mape: float = -1.0

    def predict(self) -> np.array:
        raise NotImplementedError("predict method not implemented")

    def __str__(self):
        if self.model is None:
            return "Model not trained yet"
        return (
            f"Model: {self.model_name}\n"
            f"Train values: {self.train_values}\n"
            f"Test values: {self.test_values}\n"
            f"Predictions: {self.predictions}\n"
            f"Mean Absolute Percentage Error: {self.mape}"
        )
