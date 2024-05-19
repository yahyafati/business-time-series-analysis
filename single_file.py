import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import time
import os
from functools import wraps

from dataclasses import dataclass

import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.float_format", "{:.2f}".format)

UNIQUE_ID = "id"
SEPARATOR = "::"


def mean_absolute_percentage_error(y_true: list[float], y_pred: list[float]) -> float:
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def create_unique_id(df: pd.DataFrame, n: int) -> pd.DataFrame:
    columns = list(df.columns)
    columns = columns[:n]
    df[UNIQUE_ID] = df[columns].apply(
        lambda row: SEPARATOR.join(row.values.astype(str)), axis=1
    )
    cols = list(df.columns)
    cols.insert(0, cols.pop(cols.index(UNIQUE_ID)))
    df = df[cols]
    df = df.drop(columns=columns, axis=1)
    return df


def general_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index(UNIQUE_ID)
    df = df.transpose()
    df = df.reset_index()
    df = df.rename(columns={"index": "year"})
    return df


def process_single_series(single_series):
    data = single_series.copy()
    data.columns = ["date", "rate"]
    data["date"] = pd.to_datetime(data["date"], format="%Y")
    data["rate"] = data["rate"].astype(str)
    data["rate"] = data.rate.replace("ND", np.nan)
    data["rate"] = pd.to_numeric(data["rate"])
    data = data.resample("YS", on="date", label="left", closed="left").mean()

    # gradualize the data by linearly interpolating the data
    data = data.resample("D").asfreq()
    data = data.interpolate(method="linear")

    data["time"] = [i + 1 for i in range(len(data))]

    return data


from dataclasses import dataclass


@dataclass
class ModelResult:
    model: object
    train_values: np.array
    test_values: np.array
    predictions: np.array
    mape: float


def train_test_split(data, test_ratio=0.2):
    train_size = int(len(data) * (1 - test_ratio))
    train = data[:train_size]
    test = data[train_size:]

    x_train = train.drop("rate", axis=1)
    x_test = test.drop("rate", axis=1)
    y_train = train[["rate"]]
    y_test = test[["rate"]]

    return x_train, x_test, y_train, y_test


from sklearn.linear_model import LinearRegression


def get_linear_regression_prediction(data: pd.DataFrame, test_ratio=0.2) -> ModelResult:
    x_train, x_test, y_train, y_test = train_test_split(data, test_ratio)

    model = LinearRegression()
    model.fit(X=x_train, y=y_train)
    predictions = model.predict(x_test)

    predictions = predictions.reshape(-1)

    linear_reg_mape = mean_absolute_percentage_error(
        y_test.values.reshape(-1), predictions
    )

    return ModelResult(
        model,
        y_train.values.reshape(-1),
        y_test.values.reshape(-1),
        predictions,
        linear_reg_mape,
    )


from statsmodels.tsa.arima.model import ARIMA


def get_arima_prediction(data, test_ratio=0.2, order=(3, 1, 2)) -> ModelResult:
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

    return ModelResult(
        arima,
        train.rate.values.reshape(-1),
        test.rate.values.reshape(-1),
        predictions,
        arima_mape,
    )


from prophet import Prophet


@dataclass
class ProphetModelResult(ModelResult):
    forecast: pd.DataFrame


def get_prophet_prediction(
    data, test_ratio=0.2, interval_width=0.95
) -> ProphetModelResult:
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

    return ProphetModelResult(
        my_model,
        train.y.values.reshape(-1),
        test.y.values.reshape(-1),
        predictions,
        prophet_mape,
        forecast,
    )


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def add_rows(data: pd.DataFrame, n: int) -> pd.DataFrame:
    added_df: pd.DataFrame = data.copy()
    last_row = added_df.index[-1]
    for _ in range(n):
        last_row = last_row + DateOffset(days=1)
        added_df.loc[last_row] = np.nan

    added_df = added_df.interpolate(method="linear")
    return added_df


def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    # for i in range(len(dataset)-look_back-1):
    for i in range(len(dataset) - look_back):
        a = dataset[i : (i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)
    return dataset, scaler


def split_lstm_train_test(data, test_ratio=0.2, look_back=1):
    train_size = int(len(data) * (1 - test_ratio))
    train, test = data[0:train_size, :], data[train_size : len(data), :]

    # reshape into X=t and Y=t+1
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

    return train_x, train_y, test_x, test_y


def get_trained_model(train_x, train_y, look_back, epochs=5):
    model = Sequential()

    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(train_x, train_y, epochs=epochs, batch_size=1, verbose=2)

    return model


def get_lstm_prediction(data, epochs=5, look_back=1, test_ratio=0.2) -> ModelResult:
    data = add_rows(data, look_back)
    dataset = data.rate.values
    dataset = dataset.astype("float32").reshape(-1, 1)

    dataset, scaler = normalize_data(dataset)
    train_x, train_y, test_x, test_y = split_lstm_train_test(
        dataset, test_ratio, look_back
    )

    # create and fit the LSTM network
    model = get_trained_model(train_x, train_y, look_back, epochs=epochs)

    # make predictions
    prediction = model.predict(test_x)

    # invert normalization
    train_y = scaler.inverse_transform([train_y])
    test_y = scaler.inverse_transform([test_y])
    prediction = scaler.inverse_transform(prediction)

    prediction = prediction.reshape(-1)

    lstm_mape = mean_absolute_percentage_error(test_y.reshape(-1), prediction)

    return ModelResult(
        model, train_y.reshape(-1), test_y.reshape(-1), prediction, lstm_mape
    )


def train_all_models(data, test_ratio=0.2):
    test_size = int(len(data) * test_ratio)
    train, test = data[:-test_size], data[-test_size:]

    # print("Linear Regression Model Started")
    # linear_regression_result = get_linear_regression_prediction(data, test_ratio=test_ratio)

    print("\n\nARIMA Prediction Started")
    arima_result = get_arima_prediction(data, test_ratio=test_ratio)

    print("\n\nProphet Prediction Started")
    prophet_result = get_prophet_prediction(data, test_ratio=test_ratio)

    print("\n\nLSTM Prediction Started")
    lstm_result = get_lstm_prediction(data, test_ratio=test_ratio, epochs=2)

    return {
        "test": test,
        "train": train,
        "models": {
            # "Linear Regression": linear_regression_result,
            "Prophet": prophet_result,
            "ARIMA": arima_result,
            "LSTM": lstm_result,
        },
    }


# @title Split the index to separate columns


def split_unique_ids(df, n, original_columns):
    indexes = list(df.index.str.split(SEPARATOR, n=n, expand=True))
    columns = original_columns[:n]
    split_df = df.copy()
    split_df[columns] = indexes

    columns = columns + list(df.columns)  # To Order them properly
    return split_df[columns].reset_index(drop=True)


def perform_imputation(df, imputer, n, file_name=None):
    imputed_values = imputer.fit_transform(df[df.columns[1:]])
    imputed_data = df.copy()
    imputed_data[df.columns[1:]] = imputed_values

    if file_name:
        imputed_data = imputed_data.set_index(UNIQUE_ID)
        imputed_data = split_unique_ids(imputed_data, n, ORIGINAL_COLUMNS)
        imputed_data.to_csv(file_name, sep=";", index=False)
        imputed_data = create_unique_id(imputed_data, n)
        imputed_data = imputed_data.reset_index()

    return imputed_data


# END TO END PIPELINE
from sklearn.impute import KNNImputer
import time as time_module


def main():
    start_time = time_module.time()
    # print the date time at the start of the execution
    print(f"Start time: {time_module.ctime()}")
    # Load the dataset
    dataset_url = "data.csv"
    NUMBER_OF_ID_COLS = 2  # Number of columns to be used for creating unique id
    df = pd.read_csv(dataset_url, delimiter=";")
    ORIGINAL_COLUMNS = list(df.columns)

    # Preprocess the dataset
    df = create_unique_id(df, NUMBER_OF_ID_COLS)
    print("Data Loaded!")

    ## Handle Missing Values
    df = perform_imputation(df, KNNImputer(n_neighbors=5), NUMBER_OF_ID_COLS)

    df = general_preprocessing(df)

    until_year = 2019
    size = df[pd.to_datetime(df["year"]).dt.year > until_year].shape[0]
    test_ratio = size / df.shape[0]
    print(f"Test ratio: {test_ratio}\n")

    unique_ids = list(df.columns[1:])
    final_df = pd.DataFrame()

    # make a folder called predictions if it does not exist
    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    size = len(unique_ids)
    for current_index, unique_id in enumerate(unique_ids[:2]):
        # Clear screen
        os.system("cls" if os.name == "nt" else "clear")

        print(
            f"[{time_module.ctime()}] - Processing {current_index+1}/{size}. Unique ID: {unique_id}"
        )
        # Prepare the data
        data = df[["year", unique_id]]
        data = process_single_series(data)

        # Train all models
        results = train_all_models(data, test_ratio=test_ratio)

        # Post process the results
        _train = results["train"]
        test = results["test"]
        predictions = results["models"]

        ## Get the predictions in the same size by padding with NaN
        max_size = max([pred.predictions.shape[0] for _, pred in predictions.items()])
        same_size_predictions = {}

        for model_name, model_result in predictions.items():
            model_result.predictions = np.pad(
                model_result.predictions,
                (0, max_size - len(model_result.predictions)),
                "constant",
                constant_values=(np.nan),
            )
            same_size_predictions[model_name] = model_result.predictions

        ## Create a DataFrame with the predictions
        pred_df = pd.DataFrame(same_size_predictions)
        pred_df["year"] = test.index
        pred_df["year"] = pd.to_datetime(pred_df["year"])
        pred_df = pred_df.set_index("year")

        ## Get the predictions for the first day of the years
        pred_df = pred_df[pred_df.index.dayofyear == 1]

        pred_df.index = pred_df.index.year

        test_first_day_only = test[test.index.dayofyear == 1]
        test_first_day_only.index = test_first_day_only.index.year
        for col in pred_df.columns:
            pred_df[f"{col}_MAPE"] = (
                np.abs(pred_df[col] - test_first_day_only.loc[pred_df.index]["rate"])
                / test_first_day_only.loc[pred_df.index]["rate"]
            ) * 100

        pred_df_unstacked = pred_df.unstack()
        pred_df_unstacked.index = [
            f"{year}_{model}" for model, year in pred_df_unstacked.index
        ]
        pred_df = pred_df_unstacked.to_frame().T
        pred_df.index = [unique_id]

        real_df: pd.DataFrame = df[[unique_id, "year"]]
        real_df = real_df.set_index("year")
        real_df = real_df.T

        full_column_df = pd.concat([real_df, pred_df], axis=1)
        final_df = pd.concat([final_df, full_column_df])

        # Save every 50th iteration
        if (current_index + 1) % 50 == 0:
            final_df = split_unique_ids(final_df, NUMBER_OF_ID_COLS, ORIGINAL_COLUMNS)
            final_df.to_csv(
                f"predictions/predictions_{current_index + 1}.csv", sep=";", index=False
            )

    sorted_columns = sorted(final_df.columns, key=lambda col: tuple(col.split("_")))
    final_df = final_df[sorted_columns]

    final_df = split_unique_ids(final_df, NUMBER_OF_ID_COLS, ORIGINAL_COLUMNS)
    final_df.to_csv("predictions/predictions.csv", sep=";", index=False)

    end_time = time_module.time()
    print(f"End time: {time_module.ctime()}")
    print(f"Execution time: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    main()
