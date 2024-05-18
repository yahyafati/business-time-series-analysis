from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Local imports
from utils import timed_function, mean_absolute_percentage_error
from trainers.TrainingAdapter import TrainerAdapter
from statsmodels.tsa.arima.model import ARIMA


class ARIMAAdapter(TrainerAdapter):

    def __init__(self, data: pd.DataFrame, test_ratio=0.2, order=(3, 1, 2)):
        self.order = order
        super().__init__("ARIMA", data, test_ratio)

    @timed_function("ARIMA Prediction")
    def predict(self) -> np.array:
        data = self.data.copy()
        test_ratio = self.test_ratio
        order = self.order

        train_size = int(len(data) * (1 - test_ratio))
        train = data[:train_size]
        test = data[train_size:]

        predictions = []

        arima = ARIMA(train.rate, order=order).fit()

        horizon = len(test)
        predictions.append(arima.forecast(horizon))
        predictions = np.array(predictions[0]).reshape((horizon,))

        predictions = predictions.reshape(-1)

        arima_mape = mean_absolute_percentage_error(test.rate, predictions)

        self.model = arima
        self.train_values = train.rate.values.reshape(-1)
        self.test_values = test.rate.values.reshape(-1)
        self.predictions = predictions
        self.mape = arima_mape

        return predictions
