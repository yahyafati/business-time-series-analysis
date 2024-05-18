import numpy as np
import pandas as pd
import os

# Local imports
from utils import transposer
from trainers import LinearReg, ARIMA, ProphetReg, TrainingAdapter, LSTMReg

pd.set_option("display.float_format", "{:.2f}".format)
MAPE_ROW_NAME = "MAPE"

# hide warnings
import warnings

warnings.filterwarnings("ignore")


def process_single_series(single_series):
    data = single_series.copy()
    data.columns = ["date", "rate"]
    data["rate"] = data["rate"].astype(str)

    data["rate"] = data.rate.replace("ND", np.nan)

    data["date"] = pd.to_datetime(data["date"], format="%Y")
    data["rate"] = pd.to_numeric(data["rate"])

    # handling missing values (interpolation)
    data["rate"] = data["rate"].interpolate()

    data = data.resample("YS", on="date", label="left", closed="left").mean()

    # gradualize the data by linearly interpolating the data
    data = data.resample("D").asfreq()
    data = data.interpolate(method="linear")

    data["time"] = [i + 1 for i in range(len(data))]

    return data


def get_predictions(
    data, unique_id, test_ratio=0.2
) -> list[TrainingAdapter.TrainerAdapter]:
    df_cur = data[["year", unique_id]]
    df_cur = process_single_series(df_cur)

    linear_reg = LinearReg.LinearRegAdapter(df_cur, test_ratio=test_ratio)
    arima_reg = ARIMA.ARIMAAdapter(df_cur, test_ratio=test_ratio)
    prophet_reg = ProphetReg.ProphetAdapter(df_cur, test_ratio=test_ratio)
    lstm_reg = LSTMReg.LSTMAdapter(df_cur, epochs=1, test_ratio=test_ratio)

    return [linear_reg, arima_reg, prophet_reg, lstm_reg]


def get_predictions_dataframe(
    results: list[TrainingAdapter.TrainerAdapter],
) -> pd.DataFrame:
    if len(results) == 0:
        return pd.DataFrame()

    predictions = {
        "year": results[0].future_dates,
    }

    # LSTM predictions are not of the same length as the other predictions
    # so we need to make them the same length
    max_len = max([len(predictions[key]) for key in predictions.keys()])
    for result in results:
        len_diff = max_len - len(result.predictions)
        preds = result.predictions
        if len_diff > 0:
            preds = np.append(preds, [np.nan] * len_diff)
        predictions[result.name] = preds

    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.set_index("year")

    return predictions_df


def filter_year_start(df):
    return df[df.index.dayofyear == 1]


def create_output_dataframe(df, models, unique_id):
    pred_df = get_predictions_dataframe(models)
    mape = [result.mape for result in models]
    pred_df = filter_year_start(pred_df)
    pred_df.index = pd.to_datetime(pred_df.index, format="%Y").strftime("%Y")
    pred_df.loc[MAPE_ROW_NAME] = mape

    unstacked_pred_df = pred_df.unstack()
    unstacked_pred_df.index = [
        f"{model}_{year}" for model, year in unstacked_pred_df.index
    ]
    columns = list(unstacked_pred_df.index)
    pred_df = unstacked_pred_df.to_frame().T
    pred_df.columns = columns
    pred_df.index = [unique_id]

    real_df: pd.DataFrame = df[[unique_id, "year"]]
    real_df = real_df.set_index("year")
    real_df = real_df.T
    all_df = pd.concat([real_df, pred_df], axis=1)

    return all_df


def run(file_path: str, n: int) -> pd.DataFrame:
    absolute_path = os.path.abspath(file_path)
    df = transposer.read_tranpose_data(absolute_path, n)
    df = df.reset_index()
    df = df.rename(columns={"index": "year"})

    until_year = 2019
    size = df[pd.to_datetime(df["year"]).dt.year > until_year].shape[0]
    test_ratio = size / df.shape[0]
    print(f"Test ratio: {test_ratio}\n")

    # get columns that don't have any missing values
    df = df.dropna(axis=1, how="any")

    unique_ids = list(df.columns[1:])

    final_df = pd.DataFrame()
    total = len(unique_ids)
    for i, unique_id in enumerate(unique_ids):
        models = get_predictions(df, unique_id)
        for model in models:
            model.predict()
        output_df = create_output_dataframe(df, models, unique_id)
        final_df = pd.concat([final_df, output_df])
        print(f"{i+1}/{total} - {unique_id}")

        if i == 5:
            break

    return final_df


if __name__ == "__main__":
    main()
