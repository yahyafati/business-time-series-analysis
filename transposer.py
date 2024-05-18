import numpy as np
import pandas as pd
import os

COUNTRY_SECTOR = "country_sector"
PRODUCT_SECTOR = "Product/Sector"
REPORTING_ECONOMY = "Reporting Economy"

ALL_COLUMNS = ['2005', '2006', '2007', '2008', '2009', '2010',
       '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
       '2020', '2021', '2022']

def combine_country_sector(df: pd.DataFrame) -> pd.DataFrame:
    SECTORS_IDS = df[PRODUCT_SECTOR].str.split("-").str[1].str.strip()
    df[COUNTRY_SECTOR] = df[REPORTING_ECONOMY] + "_" + SECTORS_IDS
    cols = list(df.columns)
    cols.insert(0, cols.pop(cols.index(COUNTRY_SECTOR)))
    df = df[cols]
    df = df.drop([REPORTING_ECONOMY, PRODUCT_SECTOR], axis=1)
    return df

def rename_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get rows with duplicates i.e same country and sector
    
    In this dataset, it only happens with EU countries. There are 3 rows for each sector
    
    One is the sum of the other two. We will drop the sum row, and rename the other two
    """

    df_grouped = df.groupby(COUNTRY_SECTOR)
    df_grouped = df_grouped.filter(lambda x: len(x) > 1)

    unique_country_sectors = df_grouped.country_sector.unique()
    df = df.copy()
    for country_sector in unique_country_sectors:
        rows = df_grouped[df_grouped[COUNTRY_SECTOR] == country_sector]
        rows = rows.sort_values(by=["2005", "2006", "2007", "2022"], ascending=False)
        for i, row_index in enumerate(rows.index):
            if i == 0:
                df = df.drop(row_index)
            else:
                df.loc[row_index, COUNTRY_SECTOR] = f"{country_sector}_{i+1}"
    return df

def tranpose_data(src: str, dest: str = None) -> pd.DataFrame:
    df = pd.read_csv(src, delimiter=';')
    df = combine_country_sector(df)
    
    no_of_duplicates = df[COUNTRY_SECTOR].shape[0] - df[COUNTRY_SECTOR].unique().shape[0]
    print(f"There are {no_of_duplicates} rows with same Country and Sector.")

    df = rename_duplicates(df)
    df = df.set_index(COUNTRY_SECTOR)
    df = df.transpose()

    if dest:
        df.to_csv(dest, index=True)

    return df


if __name__ == '__main__':
    print(tranpose_data("data.csv", "transposed.ignore.csv").head())