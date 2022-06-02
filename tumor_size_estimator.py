from os import path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, \
    AdaBoostRegressor
from preprocessing import load_data_question_2, DATA_PATH, TRAIN_FILE, LABELS_FILE_2

LABEL_TITLE = "Tumor Size"  # TODO check the right label header
PRED_FILE = "prediction.csv"  # TODO make sure it doesn't overrun the other part's predictions!
# TODO also address the gold part...

def estimate_tumor_size(train_X, train_y, test_X):
    ada = AdaBoostRegressor()
    ada.fit(train_X, train_y)
    pred = ada.predict(test_X)
    return pred


def main():
    X, y = load_data_question_2(path.join(DATA_PATH, TRAIN_FILE), path.join(DATA_PATH, LABELS_FILE_2))
    test_X, _ = load_data_question_2(path.join(DATA_PATH, "test.feats.csv"))
    predictions = estimate_tumor_size(X, y, test_X)

    pd.DataFrame(predictions).to_csv(PRED_FILE)

if __name__ == "__main__":
    main()

