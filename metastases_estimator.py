from os import path
import numpy as np
import pandas as pd
import sklearn

from preprocessing import load_data, DATA_PATH, TRAIN_FILE
from utils import split_train_test

LABEL_OPTIONS = ['BON - Bones', 'LYM - Lymph nodes', 'HEP - Hepatic',
                 'PUL - Pulmonary', 'PLE - Pleura', 'SKI - Skin',
                 'OTH - Other', 'BRA - Brain', 'MAR - Bone Marrow',
                 'PER - Peritoneum', 'ADR - Adrenals']


def estimate_location(location, train_X, train_y, test_X, test_y):
    return []


def main():
    X, y = load_data(path.join(DATA_PATH, TRAIN_FILE))
    train_X, train_y, test_X, test_y = split_train_test(X, y)
    results_dict = {}
    for loc in LABEL_OPTIONS:
        results_dict[loc] = estimate_location(loc, train_X, train_y, test_X, test_y)




if __name__ == "__main__":
    main()
