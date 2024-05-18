from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# Local imports
from utils import timed_function, mean_absolute_percentage_error
from trainers.TrainingAdapter import TrainerAdapter
from prophet import Prophet


class ProphetAdapter(TrainerAdapter):

    def __init__(self, data: pd.DataFrame, test_ratio=0.2, interval_width=0.95):
        super().__init__("Prophet", data, test_ratio)

        self.interval_width = interval_width
        self.forecast = None

    @timed_function("Prophet Prediction")
    def predict(self):
        data = self.data.copy()
        test_ratio = self.test_ratio
        interval_width = self.interval_width

        data = data.reset_index()
        data = data.rename(columns={"date": "ds", "rate": "y"})
        data = data[["ds", "y"]]

        train_size = int(len(data) * (1 - test_ratio))
        train = data[:train_size]
        test = data[train_size:]

        my_model = Prophet(interval_width=interval_width)
        my_model.fit(train)

        future_dates = my_model.make_future_dataframe(periods=len(test), freq="D")

        forecast = my_model.predict(future_dates)

        predictions = forecast.yhat[train_size:].values.reshape(-1)

        prophet_mape = mean_absolute_percentage_error(test.y, predictions)

        self.model = my_model
        self.train_values = train.y.values.reshape(-1)
        self.test_values = test.y.values.reshape(-1)
        self.predictions = predictions
        self.mape = prophet_mape

        self.forecast = forecast

        return predictions
