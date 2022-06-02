from os import path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

DATA_PATH = "Mission 2 - Breast Cancer"
TRAIN_FILE = "train.feats.csv"

DATES_COLS = ["אבחנה-Diagnosis date", "אבחנה-Surgery date1",
              "אבחנה-Surgery date2", "אבחנה-Surgery date3"]
SUR_SUM_COL = "אבחנה-Surgery sum"
SUR_DATES_COL = ["אבחנה-Surgery date1", "אבחנה-Surgery date2", "אבחנה-Surgery date3"]
SUR_SINCE_LAST = "Time since last surgery"


def surgery_process(df):
    # Surgery sum
    df[SUR_SUM_COL].replace({None: "0"}, inplace=True)

    # Dates
    # for col in SUR_DATES_COL:
    #     df[col] = pd.to_datetime(df[col])
    #     #df[col].replace({None: "0"}, inplace=True)
    # df[SUR_SINCE_LAST] = df[SUR_DATES_COL].max(axis=1)
    # print(df[SUR_SINCE_LAST])


def load_data(file_path):
    df = pd.read_csv(file_path, dtype='unicode', parse_dates=DATES_COLS, dayfirst=True)
    surgery_process(df)

    return df


def main():
    data = load_data(path.join(DATA_PATH, TRAIN_FILE))





if __name__ == "__main__":
    main()
