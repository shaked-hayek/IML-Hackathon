from os import path
import numpy as np
import pandas as pd
import sklearn

from preprocessing import load_data, DATA_PATH, TRAIN_FILE
from utils import split_train_test

GOLD_FILE = "gold.csv"
PRED_FILE = "pred.csv"

LABEL_TITLE = "אבחנה-Location of distal metastases"
LABEL_OPTIONS = ['BON - Bones', 'LYM - Lymph nodes', 'HEP - Hepatic',
                 'PUL - Pulmonary', 'PLE - Pleura', 'SKI - Skin',
                 'OTH - Other', 'BRA - Brain', 'MAR - Bone Marrow',
                 'PER - Peritoneum', 'ADR - Adrenals']


def estimate_location(location, train_X, train_y, test_X, test_y):
    return test_y # TEMP!


def create_output_file(results_dict):
    df = pd.DataFrame(data=results_dict)
    cols = df.columns.values
    mask = df.gt(0.0).values
    return [cols[x].tolist() for x in mask]


def labels_to_categorical(train_y):
    df = pd.DataFrame()
    for col in LABEL_OPTIONS:
        df[col] = np.where(col in train_y[LABEL_TITLE], 1, 0)
    return df


def main():
    X, y = load_data(path.join(DATA_PATH, TRAIN_FILE))
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    train_y_as_dummies = labels_to_categorical(train_y)

    results_dict = {}
    for loc in LABEL_OPTIONS:
        results_dict[loc] = estimate_location(
            loc, train_X, train_y_as_dummies[loc], test_X, test_y) # TODO: remove test_y
    pred_output = create_output_file(results_dict)

    # Write results to CSV
    with open(PRED_FILE, "w") as f:
        f.write(LABEL_TITLE + "\n")
        f.write("\n".join(pred_output))

    with open(GOLD_FILE, "w") as f:
        f.write(LABEL_TITLE + "\n")
        f.write("\n".join(test_y))

    # Mission 2 - Breast Cancer/evaluate_part_0.py --gold test_y --pred {}


if __name__ == "__main__":
    main()
