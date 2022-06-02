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

FORM_NAME = ' Form Name'
USER_NAME = "User Name"
AGE = "אבחנה-Age"
BASIC_STAGE = "אבחנה-Basic stage"
CLINICAL_STAGE = "c - Clinical"
PATHOLOGICAL_STAGE = "p - Pathological"
RECCURENT_STAGE = "r - Reccurent"
NULL = "Null"
DIAGNOSIS_DATE = "אבחנה-Diagnosis date"
NUM_OF_DAYS_SINCE_DIAGNOSIS = "Number of days"
TODAY = "today"

STAGE = "אבחנה-Stage"
STAGE_DICT = {"stage0": 0, "stage1": 1, "LA": 1, "stage1a": 1,
              "stage1b": 2, "stage2a": 3, "stage2b": 4, "stage 3a": 5,
              "stage 3b": 6, "stage 3c": 7, "stage 4": 8,
              "Not yet Established": 0, None: 0}

TUMOR_MARKERS_DIAGNOSIS = ["אבחנה-er", "אבחנה-pr"]



def surgery_process(df):
    # Surgery sum
    df[SUR_SUM_COL].replace({None: "0"}, inplace=True)

    # Dates
    for col in SUR_DATES_COL:
        df[col].replace({"Unknown": None}, inplace=True)
        df[col] = (df[TODAY] - pd.to_datetime(df[col], dayfirst=True)).dt.days
    df[SUR_SINCE_LAST] = df[SUR_DATES_COL].max(axis=1)
    df.drop(columns=SUR_DATES_COL, inplace=True)



def er_pr_preprocess(cell_data):
    """
    # TODO document
    """
    cell_data = cell_data.lower()
    if cell_data[:2] == "<1": # "<1" or "<1%"
        return 0.005
    if cell_data[:2] == ">7": # ">75%"
        return 0.8
    elif cell_data[:2] == "po":
        # check if weak
            # if so check if there is percenatage

    #### CONTINUE!!!!






def names_and_age_process(df):
    # Form Name
    df[FORM_NAME] = df[FORM_NAME].astype('category')
    df[FORM_NAME] = pd.factorize(df[FORM_NAME])[0] + 1
    # User Name
    df[USER_NAME] = (df[USER_NAME].str.split("_").str[0]).astype(int)
    # Age
    df[AGE] = df[AGE].astype(int)


def load_data(file_path):
    df = pd.read_csv(file_path, dtype='unicode')
    df[TODAY] = pd.to_datetime(TODAY)
    surgery_process(df)

    # Basic Stage
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([CLINICAL_STAGE], 1)
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([PATHOLOGICAL_STAGE], 2)
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([RECCURENT_STAGE], 3)
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([NULL], 0)

    # Stage
    df[STAGE].replace(STAGE_DICT, inplace=True)

    # Diagnosis date
    df[DIAGNOSIS_DATE] = pd.to_datetime(df[DIAGNOSIS_DATE])
    df['Difference'] = (df[TODAY] - df[DIAGNOSIS_DATE]).dt.days

    # Markers (er & pr)
    for marker in TUMOR_MARKERS_DIAGNOSIS:
        df[marker] = df[marker].astype(str).apply(er_pr_preprocess)

    return df


def main():
    data = load_data(path.join(DATA_PATH, TRAIN_FILE))


if __name__ == "__main__":
    main()
