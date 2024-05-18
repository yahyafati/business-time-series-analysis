from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Local imports
from utils import timed_function, mean_absolute_percentage_error
from . import TrainerAdapter


class LinearRegAdapter(TrainerAdapter):

    def __init__(self, data: pd.DataFrame, test_ratio=0.2):
        super().__init__("Linear Regression", data, test_ratio)

    def __train_test_split(self, data):
        test_ratio = self.test_ratio
        train_size = int(len(data) * (1 - test_ratio))
        train = data[:train_size]
        test = data[train_size:]

        x_train = train.drop("rate", axis=1)
        x_test = test.drop("rate", axis=1)
        y_train = train[["rate"]]
        y_test = test[["rate"]]

        return x_train, x_test, y_train, y_test

    @timed_function
    def predict(self) -> np.array:
        data = self.data.copy()
        x_train, x_test, y_train, y_test = self.__train_test_split(data)

        model = LinearRegression()
        model.fit(X=x_train, y=y_train)
        predictions = model.predict(x_test)

        predictions = predictions.reshape(-1)

        linear_reg_mape = mean_absolute_percentage_error(
            y_test.values.reshape(-1), predictions
        )

        self.model = model
        self.train_values = y_train.values.reshape(-1)
        self.test_values = y_test.values.reshape(-1)
        self.predictions = predictions.reshape(-1)
        self.mape = linear_reg_mape

        return predictions
