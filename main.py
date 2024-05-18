import numpy as np
import pandas as pd
import os

# Local imports
from utils import transposer
from trainers import LinearReg


def process_country_sector(column_sector):
    data = column_sector.copy()
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


def predict_country_sector(data, country_sector):
    country_sector_data = data[["year", country_sector]]
    country_sector_data = process_country_sector(country_sector_data)

    linear_reg = LinearReg.LinearRegAdapter(country_sector_data)
    result = linear_reg.predict()

    return result


def main():
    file_path = "data.csv"
    absolute_path = os.path.abspath(file_path)
    df = transposer.read_tranpose_data(absolute_path)
    df = df.reset_index()
    df = df.rename(columns={"index": "year"})

    result = predict_country_sector(df, "United States of America_MA")

    print("Enter 'q' to quit")
    q = input(">> ")
    while q != "q":
        res = None
        try:
            res = eval(q)
        except Exception as e:
            res = e
        if res is not None:
            print(res)
        q = input(">> ")


if __name__ == "__main__":
    main()
