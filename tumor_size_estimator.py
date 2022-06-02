from os import path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from preprocessing import load_data, DATA_PATH, TRAIN_FILE, LABELS_FILE_2

LABEL_TITLE = "Tumor Size"  # TODO check the right label header
PRED_FILE = "prediction.csv"  # TODO make sure it doesn't overrun the other part's predictions!
# TODO also address the gold part...


def main():
    train_X, train_y = load_data(path.join(DATA_PATH, TRAIN_FILE))
    part2_model = RandomForestRegressor()  # only temporary #TODO choose the right one
    part2_model.fit(train_X, train_y)
    test_X, _ = load_data(path.join(DATA_PATH, LABELS_FILE_2))
    pd.DataFrame(part2_model.predict(test_X), columns=[LABEL_TITLE]).to_csv(PRED_FILE, index=False)


if __name__ == "__main__":
    main()

