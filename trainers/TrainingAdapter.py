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
            "-" * 50 + "\n"
            f"Model: {self.model_name}\n"
            f"Mean Absolute Percentage Error: {self.mape}" + "\n" + "-" * 50
        )

    def __repr__(self):
        return f"{self.model_name} Model - MAPE: {self.mape:.2f}"
