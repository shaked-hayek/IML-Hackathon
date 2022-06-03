from os import path

import numpy as np
import pandas as pd

DATA_PATH = "Mission 2 - Breast Cancer"
TRAIN_FILE = "train.feats.csv"
TEST_FILE = "test.feats.csv"
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
POS_NODES = "אבחנה-Positive nodes"
EXAM_NODES = "אבחנה-Nodes exam"
ID_COL = "id-hushed_internalpatientid"

DROP = [" Hospital", "אבחנה-Her2", "אבחנה-Histopatological degree"
    , "אבחנה-Ivi -Lymphovascular invasion", "אבחנה-KI67 protein", "אבחנה-Lymphatic penetration",
        "אבחנה-M -metastases mark (TNM)", "אבחנה-Margin Type", "אבחנה-N -lymph nodes mark (TNM)", "אבחנה-Surgery name1",
        "אבחנה-Surgery name2", "אבחנה-Surgery name3",
        "אבחנה-T -Tumor mark (TNM)", "אבחנה-Tumor depth", "אבחנה-Tumor width",
        "surgery before or after-Actual activity",
        "surgery before or after-Activity date", "אבחנה-er", "אבחנה-pr"
        ]

TODAY = "today"

STAGE = "אבחנה-Stage"
STAGE_DICT = {"Stage0": 0, "Stage0is": 0, "Stage0a": 0, "Stage1": 1, "LA": 1, "Stage1a": 1,
              "Stage1b": 2, "Stage1c": 3, "Stage2": 4, "Stage2a": 4, "Stage2b": 5, "Stage3": 6, "Stage3a": 6,
              "Stage 3a": 6, "Stage 3b": 7, "Stage3b": 7, "Stage 3c": 8, "Stage3c": 8,
              "Stage 4": 9, "Stage4": 9, "Not yet Established": 0}

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
    cell_data = cell_data.lower()
    if len(cell_data) == 0:
        return 0
    if len(cell_data) == 1:
        return
    elif cell_data[:2] == "<1":  # "<1" or "<1%"
        return 0.005
    if cell_data[:2] == ">7":  # ">75%"
        return 0.8
    elif cell_data[:2] == "po":
        return




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
    df[NUM_OF_DAYS_SINCE_DIAGNOSIS] = (df[TODAY] - df[DIAGNOSIS_DATE]).dt.days
    df.drop(columns=DIAGNOSIS_DATE, inplace=True)
    df.drop(columns=TODAY, inplace=True)
    # diagnosis
    df[DIAGNOSIS] = df[DIAGNOSIS].astype('category')
    df[DIAGNOSIS] = pd.factorize(df[DIAGNOSIS])[0] + 1

    # # Markers (er & pr) - can be returned from comment only if we had time
    #                   #   finishing the er_pr_preprocess function
    # for marker in TUMOR_MARKERS_DIAGNOSIS:
    #     df[marker] = df[marker].astype(str).apply(er_pr_preprocess)


def nodes_process(df):
    # Ratio between the amount of vertices tested versus the amount of positives
    df[EXAM_NODES].replace({None: np.nan}, inplace=True)
    df[POS_NODES].replace({None: np.nan}, inplace=True)
    df[EXAM_NODES] = df[EXAM_NODES].astype(float)
    df[POS_NODES] = df[POS_NODES].astype(float)
    df[EXAM_NODES] = df[POS_NODES] / df[EXAM_NODES]


# in classification we can use the ID but in regression the ID will add noise
def id_process(df, is_question_1):
    if is_question_1:
        df[ID_COL] = df[ID_COL].astype('category')
        df[ID_COL] = pd.factorize(df[ID_COL])[0] + 1
    else:
        df.drop(columns=ID_COL, inplace=True)


def nan_process(df):
    df[POS_NODES].fillna(0, inplace=True)
    df[EXAM_NODES].fillna(0, inplace=True)
    df[SIDE].fillna(0, inplace=True)
    df[STAGE].fillna(0, inplace=True)
    df[SUR_SINCE_LAST].fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)


def load_data_question_1(file_path, labels_path=None):
    df = pd.read_csv(file_path, dtype='unicode')
    labels = None
    if labels_path:
        labels = pd.read_csv(labels_path, dtype='unicode')
    df[LABELS_COL] = labels
    df.drop(columns=DROP, inplace=True)
    df[TODAY] = pd.to_datetime(TODAY)
    surgery_process(df)
    names_and_age_process(df)
    basic_stage_process(df)
    diagnoses_process(df)
    nodes_process(df)
    id_process(df, True)
    nan_process(df)
    labels = pd.DataFrame()
    labels[LABELS_COL] = df[LABELS_COL]
    df.drop(columns=LABELS_COL, inplace=True)
    return df, labels


def load_data_question_2(file_path, labels_path=None):
    df = pd.read_csv(file_path, dtype='unicode')
    labels = None
    if labels_path:
        labels = pd.read_csv(labels_path, dtype='unicode')
    df[LABELS_COL] = labels
    df.drop(columns=DROP, inplace=True)
    df[TODAY] = pd.to_datetime(TODAY)
    surgery_process(df)
    names_and_age_process(df)
    basic_stage_process(df)
    diagnoses_process(df)
    id_process(df, False)
    nan_process(df)
    # drop unnecessary columns for question 2
    df.drop(columns=[EXAM_NODES, POS_NODES])
    labels = df[LABELS_COL]
    df.drop(columns=LABELS_COL, inplace=True)
    return df, labels


def main():
    data1, labels1 = load_data_question_1(path.join(DATA_PATH, TRAIN_FILE), path.join(DATA_PATH, LABELS_FILE_1))
    data2, labels2 = load_data_question_2(path.join(DATA_PATH, TRAIN_FILE), path.join(DATA_PATH, LABELS_FILE_2))


if __name__ == "__main__":
    main()
