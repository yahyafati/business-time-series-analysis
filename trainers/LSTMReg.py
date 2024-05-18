import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from utils import timed_function, mean_absolute_percentage_error
from trainers.TrainingAdapter import TrainerAdapter
from prophet import Prophet


class LSTMAdapter(TrainerAdapter):

    def __init__(self, data, test_ratio=0.2, look_back=1, epochs=5):
        super().__init__("LSTM", data, test_ratio)

        self.look_back = look_back
        self.scaler = None
        self.epochs = epochs
        self.look_back = look_back

    # the side effect of this is, that it actually removes look_back number of columns
    @staticmethod
    def create_dataset(dataset, look_back=1):
        data_x, data_y = [], []
        # for i in range(len(dataset)-look_back-1):
        for i in range(len(dataset) - look_back):
            a = dataset[i : (i + look_back), 0]
            data_x.append(a)
            data_y.append(dataset[i + look_back, 0])
        return np.array(data_x), np.array(data_y)

    @staticmethod
    def split_lstm_train_test(data, test_ratio=0.2, look_back=1):
        train_size = int(len(data) * (1 - test_ratio))
        train, test = data[0:train_size, :], data[train_size : len(data), :]

        # reshape into X=t and Y=t+1
        train_x, train_y = LSTMAdapter.create_dataset(train, look_back)
        test_x, test_y = LSTMAdapter.create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

        return train_x, train_y, test_x, test_y

    @staticmethod
    def get_trained_model(train_x, train_y, look_back, epochs=5):
        model = Sequential()

        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss="mean_squared_error", optimizer="adam")
        model.fit(train_x, train_y, epochs=epochs, batch_size=1, verbose=2)

        return model

    @staticmethod
    def normalize_data(data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(data)
        return dataset, scaler

    @timed_function("LSTM Prediction")
    def predict(self):
        test_ratio = self.test_ratio
        look_back = self.look_back
        epochs = self.epochs

        dataset = self.data.rate.values

        dataset = dataset.astype("float32").reshape(-1, 1)

        dataset, scaler = self.normalize_data(dataset)

        train_x, train_y, test_x, test_y = self.split_lstm_train_test(
            dataset, test_ratio, look_back
        )

        # create and fit the LSTM network
        model = self.get_trained_model(train_x, train_y, look_back, epochs=epochs)

        # make predictions
        prediction = model.predict(test_x)

        # invert normalization
        train_y = scaler.inverse_transform([train_y])
        test_y = scaler.inverse_transform([test_y])
        prediction = scaler.inverse_transform(prediction)

        prediction = prediction.reshape(-1)

        lstm_mape = mean_absolute_percentage_error(test_y.reshape(-1), prediction)

        self.scaler = scaler

        self.model = model
        self.train_values = train_y.reshape(-1)
        self.test_values = test_y.reshape(-1)
        self.predictions = prediction
        self.mape = lstm_mape

        return prediction
