import numpy as np
import pandas as pd
import os

UNIQUE_ID = "unique_id"
SEPARATOR = "_"
# PRODUCT_SECTOR = "Product/Sector"
# REPORTING_ECONOMY = "Reporting Economy"


def get_all_columns() -> list[str]:
    ALL_COLUMNS = [str(x) for x in range(2005, 2023)]
    return ALL_COLUMNS.copy()


def create_unique_id(df: pd.DataFrame, n: int) -> pd.DataFrame:
    columns = list(df.columns)
    columns = columns[:n]
    print(columns)
    df[UNIQUE_ID] = df[columns].apply(
        lambda row: SEPARATOR.join(row.values.astype(str)), axis=1
    )
    cols = list(df.columns)
    cols.insert(0, cols.pop(cols.index(UNIQUE_ID)))
    df = df[cols]
    df = df.drop(columns=columns, axis=1)
    return df


def rename_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get rows with duplicates i.e same unique_ids

    For instance,
    In this dataset, it only happens with EU countries. There are 3 rows for each sector

    One is the sum of the other two. We will drop the sum row, and rename the other two
    """

    df_grouped = df.groupby(UNIQUE_ID)
    df_grouped = df_grouped.filter(lambda x: len(x) > 1)

    unique_ids = df_grouped[UNIQUE_ID].unique()
    df = df.copy()
    for unique_id in unique_ids:
        rows = df_grouped[df_grouped[UNIQUE_ID] == unique_id]
        # TODO: This is hardcoded. We should find a way to sort the columns dynamically
        rows = rows.sort_values(by=["2005", "2006", "2007", "2022"], ascending=False)
        for i, row_index in enumerate(rows.index):
            if i == 0:
                df = df.drop(row_index)
            else:
                df.loc[row_index, UNIQUE_ID] = f"{unique_id}_{i+1}"
    return df


def tranpose_data(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = create_unique_id(df, n)
    df = rename_duplicates(df)
    df = df.set_index(UNIQUE_ID)
    df = df.transpose()

    return df


def read_tranpose_data(src: str, n: int = 2, dest: str = None) -> pd.DataFrame:
    df = pd.read_csv(src, delimiter=";")
    df = tranpose_data(df, n)

    if dest:
        df.to_csv(dest, index=True)

    return df


if __name__ == "__main__":
    print(read_tranpose_data("data.csv", "transposed.ignore.csv").head())
