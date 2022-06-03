from os import path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_data_question_1, DATA_PATH, TRAIN_FILE, \
    LABELS_FILE_1, LABELS_COL, TEST_FILE
from utils import split_train_test

GOLD_FILE = "gold.csv"
PRED_FILE = "pred.csv"

LABEL_TITLE = LABELS_COL
LABEL_OPTIONS = ['BON - Bones', 'LYM - Lymph nodes', 'HEP - Hepatic',
                 'PUL - Pulmonary', 'PLE - Pleura', 'SKI - Skin',
                 'OTH - Other', 'BRA - Brain', 'MAR - Bone Marrow',
                 'PER - Peritoneum', 'ADR - Adrenals']


def estimate_location(train_X, train_y, test_X):
    ada = RandomForestClassifier(n_estimators=100, random_state=1)
    ada.fit(train_X, train_y)
    pred = ada.predict(test_X)
    return pred


def create_output(df):
    cols = df.columns.values
    mask = df.gt(0.0).values
    out = [cols[x].tolist() for x in mask]
    out_df = pd.DataFrame([str(x) for x in out])
    return out_df.rename(columns={0: LABEL_TITLE})


def labels_to_categorical(train_y):
    df = pd.DataFrame()
    for col in LABEL_OPTIONS:
        df[col] = train_y[LABEL_TITLE].apply(lambda s: 1 if col in s else 0)
    return df


def main():
    X, y = load_data_question_1(path.join(DATA_PATH, TRAIN_FILE), path.join(DATA_PATH, LABELS_FILE_1))
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    train_y_as_dummies = labels_to_categorical(train_y)
    test_X, _ = load_data_question_1(path.join(DATA_PATH, TEST_FILE))

    results_df = pd.DataFrame()
    for loc in LABEL_OPTIONS:
        results_df[loc] = estimate_location(train_X, train_y_as_dummies[loc], test_X)
    pred_output = create_output(results_df)

    # Write results to CSV
    pred_output.to_csv(PRED_FILE, index=False)
    # test_y.to_csv(GOLD_FILE, index=False)

    # To test run:
    # python "Mission 2 - Breast Cancer/evaluate_part_0.py" --gold gold.csv --pred pred.csv


if __name__ == "__main__":
    main()
