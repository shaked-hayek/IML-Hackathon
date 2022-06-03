from os import path

import pandas as pd
from sklearn.linear_model import Ridge

from utils import split_train_test

from preprocessing import load_data_question_2, DATA_PATH, TRAIN_FILE, LABELS_FILE_2, TEST_FILE

LABELS = "labels"

LABEL_TITLE = "Tumor Size"
PRED_FILE = "prediction.csv"

def estimate_tumor_size(train_X, train_y, test_X, alpha=1.0):

    ada = Ridge(alpha=alpha)
    ada.fit(train_X, train_y)

    pred = ada.predict(test_X)
    return pred


def main():
    X, y = load_data_question_2(path.join(DATA_PATH, TRAIN_FILE), path.join(DATA_PATH, LABELS_FILE_2))

    train_X, train_y, test_X, test_y = split_train_test(X, y)
    test_X, _ = load_data_question_2(path.join(DATA_PATH, TEST_FILE))
    predictions = estimate_tumor_size(train_X, train_y, test_X, alpha=0.05)


    pd.DataFrame(predictions).rename(columns={0: LABELS}).to_csv(PRED_FILE, index=False)






if __name__ == "__main__":
    main()

