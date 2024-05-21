import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
import os
from functools import wraps

from dataclasses import dataclass

import warnings
from sklearn.impute import KNNImputer
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


@dataclass
class ModelResult:
    model: object
    train_values: np.array
    test_values: np.array
    predictions: np.array
    mape: float


def import_modules():
    global MODULES_LOADED
    if not MODULES_LOADED:
        global tf, Sequential, Dense, LSTM, Prophet, ARIMA

        from statsmodels.tsa.arima.model import ARIMA

        from prophet import Prophet

        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import LSTM

        tf.get_logger().setLevel("ERROR")

        MODULES_LOADED = True


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


def rename_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df_grouped = df.groupby(UNIQUE_ID)
    df_grouped = df_grouped.filter(lambda x: len(x) > 1)

    unique_ids = df_grouped[UNIQUE_ID].unique()
    df = df.copy()
    for unique_id in unique_ids:
        rows = df_grouped[df_grouped[UNIQUE_ID] == unique_id]
        for i, row_index in enumerate(rows.index):
            df.loc[row_index, UNIQUE_ID] = f"{unique_id}_{i+1}"
    return df


def general_preprocessing(df: pd.DataFrame, number_of_id_columns: int) -> pd.DataFrame:
    df = create_unique_id(df, number_of_id_columns)
    df = rename_duplicates(df)
    df = perform_imputation(df, KNNImputer(n_neighbors=5), number_of_id_columns)

    df = df.set_index(UNIQUE_ID)
    df = df.transpose()
    df = df.reset_index()
    df = df.rename(columns={"index": "year"})
    return df


def process_single_series(single_series):
    data = single_series.copy()
    data.columns = ["date", "rate"]
    data["date"] = pd.to_datetime(data["date"], format="%Y")
    data["rate"] = pd.to_numeric(data["rate"])
    data = data.resample("YS", on="date", label="left", closed="left").mean()

    # gradualize the data by linearly interpolating the data
    data = data.resample("D").asfreq()
    data = data.interpolate(method="linear")
    data["time"] = [i + 1 for i in range(len(data))]

    return data


def train_test_split(data, test_ratio=0.2):
    train_size = int(len(data) * (1 - test_ratio))
    train = data[:train_size]
    test = data[train_size:]

    x_train = train.drop("rate", axis=1)
    x_test = test.drop("rate", axis=1)
    y_train = train[["rate"]]
    y_test = test[["rate"]]

    return x_train, x_test, y_train, y_test


# ARIMA


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


# Prophet


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


# LSTM


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


# All Models


def train_all_models(data, test_ratio=0.2):
    test_size = int(len(data) * test_ratio)
    train, test = data[:-test_size], data[-test_size:]

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


def split_unique_ids(df, n, original_columns):
    indexes = list(df.index.str.split(SEPARATOR, n=n, expand=True))
    columns = original_columns[:n]
    split_df = df.copy()
    split_df[columns] = indexes

    columns = columns + list(df.columns)  # To Order them properly
    return split_df[columns].reset_index(drop=True)


def perform_imputation(df, imputer, n, file_name=None, sep=";"):
    imputed_values = imputer.fit_transform(df[df.columns[1:]])
    imputed_data = df.copy()
    imputed_data[df.columns[1:]] = imputed_values

    if file_name:
        imputed_data = imputed_data.set_index(UNIQUE_ID)
        imputed_data = split_unique_ids(imputed_data, n, ORIGINAL_COLUMNS)
        imputed_data.to_csv(file_name, sep=sep, index=False)
        imputed_data = create_unique_id(imputed_data, n)
        imputed_data = imputed_data.reset_index()

    return imputed_data


# END TO END PIPELINE

MODULES_LOADED = False
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
pd.set_option("display.float_format", "{:.2f}".format)

UNIQUE_ID = "id"
SEPARATOR = "::"

ERROR_IDS = []


def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timeformat = "Time Left: {:02d}:{:02d}".format(mins, secs)
        print(timeformat, end="\r")
        time.sleep(1)
        t -= 1


def train_column(df, unique_id, test_ratio=0.2):
    # Prepare the data
    data = df[["year", unique_id]]
    data = process_single_series(data)

    # Train all models
    try:
        results = train_all_models(data, test_ratio=test_ratio)
    except Exception as e:
        print(f"Error in {unique_id}: {e}")
        ERROR_IDS.append(unique_id)
        return

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
    pred_df["year"] = pd.to_datetime(test.index)
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

    return full_column_df


def read_data(file_path, sep=";"):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, sep=sep)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a csv or xlsx file.")
    return df


def get_test_ratio(df, until_year):
    size = df[pd.to_datetime(df["year"]).dt.year > until_year].shape[0]
    test_ratio = size / df.shape[0]
    return test_ratio


def main(dataset_url, number_of_id_columns, last_saved, batch_size, sep=";"):
    print("Importing modules...")
    import_modules()
    print("Modules imported successfully.\n")

    start_time = time.time()
    print(f"Start time: {time.ctime()}")

    df = read_data(dataset_url, sep=sep)
    ORIGINAL_COLUMNS = list(df.columns)

    df = general_preprocessing(df, number_of_id_columns)
    test_ratio = get_test_ratio(df, 2019)

    print(f"Test ratio: {test_ratio}\n")
    unique_ids = list(df.columns[1:])
    final_df = pd.DataFrame()

    # make a folder called predictions if it does not exist
    if not os.path.exists("predictions"):
        os.makedirs("predictions")

    size = len(unique_ids)
    cooldown_minutes = 5

    for current_index, unique_id in enumerate(unique_ids[last_saved:]):
        # Clear screen
        os.system("cls" if os.name == "nt" else "clear")

        print(
            f"[{time.ctime()}] - Processing {current_index + last_saved+1}/{size}. \nUnique ID: {unique_id}."
        )
        full_column_df = train_column(df, unique_id, test_ratio=test_ratio)
        final_df = pd.concat([final_df, full_column_df])

        # Save every 50th iteration
        current_index += last_saved + 1
        if current_index % batch_size == 0:
            final_df = split_unique_ids(
                final_df, number_of_id_columns, ORIGINAL_COLUMNS
            )
            final_df.to_csv(
                f"predictions/predictions_{current_index}.csv", sep=sep, index=False
            )
            final_df = pd.DataFrame()

        if current_index % int(batch_size) == 0:
            print(f"\n\nSleeping for {cooldown_minutes:.2f} minutes. (For CPU cooling)")
            print(
                f"Next batch will start at {time.ctime(time.time() + cooldown_minutes * 60)}"
            )
            # time.sleep(cooldown_minutes * 60)
            countdown(cooldown_minutes * 60)

    sorted_columns = sorted(final_df.columns, key=lambda col: tuple(col.split("_")))
    final_df = final_df[sorted_columns]

    if final_df.shape[0] > 0:
        final_df = split_unique_ids(final_df, number_of_id_columns, ORIGINAL_COLUMNS)
        final_df.to_csv(f"predictions/predictions_{size}.csv", sep=sep, index=False)

    end_time = time.time()
    print(f"End time: {time.ctime()}")
    print(f"Execution time: {end_time - start_time:.6f} seconds")


def combine_files(files, output_file, sep=";"):
    dfs = []
    files = list(sorted(files, key=lambda file: int(file.split("_")[-1].split(".")[0])))
    for file in files:
        df = pd.read_csv(file, sep=sep)
        dfs.append(df)

    final_df = pd.concat(dfs)
    final_df.to_csv(output_file, sep=sep, index=False)


import argparse
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers and stuff.")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="The path to the file containing the data.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2,
        help="The number of columns to be used for creating unique id.",
    )
    parser.add_argument(
        "--sep",
        "-s",
        type=str,
        default=";",
        help="The separator used in the csv file.",
    )
    parser.add_argument(
        "--last_saved",
        "-l",
        type=int,
        default=0,
        help="The number of unique ids saved in the last run.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=25,
        help="The number of unique ids to be processed before saving the results.",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s (version 0.1)",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    DATASET_URL = args.file
    NUMBER_OF_ID_COLS = args.n
    LAST_SAVED = args.last_saved
    BATCH_SIZE = args.batch_size
    SEP = args.sep

    try:
        main(DATASET_URL, NUMBER_OF_ID_COLS, LAST_SAVED, BATCH_SIZE, SEP)
        combine_files(
            [f"predictions/{file}" for file in os.listdir("predictions")],
            "predictions.csv",
            sep=SEP,
        )
    except KeyboardInterrupt:
        print("\n\n[e] - Execution stopped by the user.")
    except Exception as e:
        traceback.print_exc()
    finally:
        with open("error_ids.txt", "a") as file:
            for error_id in ERROR_IDS:
                file.write(f"{error_id}\n")

    print("\a" * 3)
