import numpy as np
import pandas as pd


class TrainerAdapter:

    def __init__(self, name: str, data: pd.DataFrame, test_ratio=0.2):
        self.name = name
        self.data = data.copy()
        self.test_ratio = test_ratio

        self.model = None
        self.train_values: np.array = None
        self.test_values: np.array = None
        self.predictions: np.array = None
        self.mape: float = -1.0

    @property
    def future_dates(self) -> np.array:
        test_size = int(len(self.data) * self.test_ratio)
        return self.data.index.values[-test_size:].reshape(-1)

    @property
    def training_dates(self) -> np.array:
        test_size = int(len(self.data) * self.test_ratio)
        return self.data.index.values[:-test_size].reshape(-1)

    @property
    def all_dates(self):
        return self.data.index.values.reshape(-1)

    def predict(self) -> np.array:
        raise NotImplementedError("predict method not implemented")

    def __str__(self):
        if self.model is None:
            return "Model not trained yet"
        return (
            "-" * 50 + "\n"
            f"Model: {self.name}\n"
            f"Mean Absolute Percentage Error: {self.mape}" + "\n" + "-" * 50
        )

    def __repr__(self):
        return f"{self.name} Model - MAPE: {self.mape:.2f}"
