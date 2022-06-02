from os import path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

DATA_PATH = "Mission 2 - Breast Cancer"
TRAIN_FILE = "train.feats.csv"

DATES_COLS = ["אבחנה-Diagnosis date", "אבחנה-Surgery date1",
              "אבחנה-Surgery date2", "אבחנה-Surgery date3"]

FORM_NAME = ' Form Name'
USER_NAME = "User Name"
AGE = "אבחנה-Age"
BASIC_STAGE = "אבחנה-Basic stage"
CLINICAL_STAGE = "c - Clinical"
PATHOLOGICAL_STAGE = "p - Pathological"
RECCURENT_STAGE = "r - Reccurent"
NULL = "Null"
DIAGNOSIS_DATE = "אבחנה-Diagnosis date"



def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=DATES_COLS, dayfirst=True)
    # parse?
    # Form Name
    df[FORM_NAME] = df[FORM_NAME].astype('category')
    df[FORM_NAME] = pd.factorize(df[FORM_NAME])[0] + 1
    # User Name
    df[USER_NAME] = (df[USER_NAME].str.split("_").str[0]).astype(int)
    # Age
    df[AGE] = df[AGE].astype(int)
    # Basic Stage
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([CLINICAL_STAGE], 1)
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([PATHOLOGICAL_STAGE], 2)
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([RECCURENT_STAGE], 3)
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([NULL], 0)
    # Diagnosis date
    df[DIAGNOSIS_DATE] = pd.to_datetime(df[DIAGNOSIS_DATE])

    return df


def main():
    data = load_data(path.join(DATA_PATH, TRAIN_FILE))





if __name__ == "__main__":
    main()
