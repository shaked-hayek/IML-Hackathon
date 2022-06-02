from os import path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

DATA_PATH = "Mission 2 - Breast Cancer"
TRAIN_FILE = "train.feats.csv"
LABELS_FILE_1 = "train.labels.0.csv"
LABELS_FILE_2 = "train.labels.1.csv"

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
NUM_OF_DAYS_SINCE_DIAGNOSIS = "Number of days since diagnosed"
DIAGNOSIS = "אבחנה-Histological diagnosis"
LABELS_COL = "labels"

DROP = [FORM_NAME, USER_NAME," Hospital", "אבחנה-Her2", "אבחנה-Histological diagnosis", "אבחנה-Histopatological degree"
        , "אבחנה-Ivi -Lymphovascular invasion", "אבחנה-KI67 protein", "אבחנה-Lymphatic penetration",
        "אבחנה-M -metastases mark (TNM)", "אבחנה-Margin Type", "אבחנה-N -lymph nodes mark (TNM)",
        "אבחנה-Side", "אבחנה-Surgery name1", "אבחנה-Surgery name2", "אבחנה-Surgery name3",
        "אבחנה-T -Tumor mark (TNM)", "אבחנה-Tumor depth", "אבחנה-Tumor width", "surgery before or after-Actual activity",
        "surgery before or after-Activity date"
        ]

NUM_OF_DAYS_SINCE_DIAGNOSIS = "Number of days"
TODAY = "today"

STAGE = "אבחנה-Stage"
STAGE_DICT = {"stage0": 0, "stage1": 1, "LA": 1, "stage1a": 1,
              "stage1b": 2, "stage2a": 3, "stage2b": 4, "stage 3a": 5,
              "stage 3b": 6, "stage 3c": 7, "stage 4": 8,
              "Not yet Established": 0, None: 0}

SIDE = "אבחנה-Side"
SIDE_DICT = {"ימין": 1, "שמאל": 1, "דו צדדי": 2}

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
        return # TODO change
        # check if weak
            # if so check if there is percenatage

    #### TODO CONTINUE!!!!


def names_and_age_process(df):
    # Form Name
    df[FORM_NAME] = df[FORM_NAME].astype('category')
    df[FORM_NAME] = pd.factorize(df[FORM_NAME])[0] + 1
    # User Name
    df[USER_NAME] = (df[USER_NAME].str.split("_").str[0]).astype(int)
    # Age
    df[AGE] = df[AGE].astype(float).round().astype(int)


def basic_stage_process(df):
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([CLINICAL_STAGE], 1)
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([PATHOLOGICAL_STAGE], 2)
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([RECCURENT_STAGE], 3)
    df[BASIC_STAGE] = df[BASIC_STAGE].replace([NULL], 0)

    # Stage
    df[STAGE].replace(STAGE_DICT, inplace=True)

    # Side
    df[SIDE].replace(SIDE_DICT, inplace=True)


def diagnoses_process(df):
    # date
    df[DIAGNOSIS_DATE] = pd.to_datetime(df[DIAGNOSIS_DATE])
    df[NUM_OF_DAYS_SINCE_DIAGNOSIS] = (df['today'] - df[DIAGNOSIS_DATE]).dt.days
    df.drop(columns=DIAGNOSIS_DATE, inplace=True)
    # diagnosis
    df[DIAGNOSIS] = df[DIAGNOSIS].astype('category')
    df[DIAGNOSIS] = pd.factorize(df[DIAGNOSIS])[0] + 1
    df['Difference'] = (df[TODAY] - df[DIAGNOSIS_DATE]).dt.days

    # Markers (er & pr)
    for marker in TUMOR_MARKERS_DIAGNOSIS:
        df[marker] = df[marker].astype(str).apply(er_pr_preprocess)


def load_data(file_path, labels_path=None, is_train=False): # TODO use the is_train flag to train only on rellevant data
    df = pd.read_csv(file_path, dtype='unicode', parse_dates=DATES_COLS, dayfirst=True)
    if (labels_path):
        labels = pd.read_csv(labels_path, dtype='unicode')
        df[LABELS_COL] = labels
    df.drop(columns=DROP, inplace=True)
    df[TODAY] = pd.to_datetime(TODAY)
    surgery_process(df)
    names_and_age_process(df)
    basic_stage_process(df)
    diagnoses_process(df)

    return (df,df)  # TODO if is_train then return also labels!!!!


def main():
    data_1, data_2 = load_data(path.join(DATA_PATH, TRAIN_FILE), LABELS_FILE_1)


if __name__ == "__main__":
    main()
