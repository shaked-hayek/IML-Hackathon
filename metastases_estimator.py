from os import path
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

from preprocessing import load_data_question_1, DATA_PATH, TRAIN_FILE, LABELS_FILE_1, LABELS_COL
from utils import split_train_test

GOLD_FILE = "gold.csv"
PRED_FILE = "pred.csv"

LABEL_TITLE = LABELS_COL
LABEL_OPTIONS = ['BON - Bones', 'LYM - Lymph nodes', 'HEP - Hepatic',
                 'PUL - Pulmonary', 'PLE - Pleura', 'SKI - Skin',
                 'OTH - Other', 'BRA - Brain', 'MAR - Bone Marrow',
                 'PER - Peritoneum', 'ADR - Adrenals']


def estimate_location(train_X, train_y, test_X):
    ada = AdaBoostClassifier()
    ada.fit(train_X, train_y)
    pred = ada.predict(test_X)
    return pred


def create_output(df):
    cols = df.columns.values
    mask = df.gt(0.0).values
    return [cols[x].tolist() for x in mask]


def labels_to_categorical(train_y):
    df = pd.DataFrame()
    for col in LABEL_OPTIONS:
        df[col] = train_y[LABEL_TITLE].apply(lambda s: 1 if col in s else 0)
    return df


def main():
    X, y = load_data_question_1(path.join(DATA_PATH, TRAIN_FILE), path.join(DATA_PATH, LABELS_FILE_1))
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    train_y_as_dummies = labels_to_categorical(train_y)

    results_df = pd.DataFrame()
    for loc in LABEL_OPTIONS:
        results_df[loc] = estimate_location(train_X, train_y_as_dummies[loc], test_X)
    pred_output = create_output(results_df)

    # Write results to CSV
    with open(PRED_FILE, "w") as f:
        f.write("\n".join([str(x) for x in pred_output]))

    with open(GOLD_FILE, "w") as f:
        f.write("\n".join(list(test_y[LABEL_TITLE])))

    # To test run:
    # python "Mission 2 - Breast Cancer/evaluate_part_0.py" --gold gold.csv --pred pred.csv


if __name__ == "__main__":
    main()
