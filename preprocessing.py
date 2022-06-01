from os import path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

DATA_PATH = "Mission 2 - Breast Cancer"
TRAIN_FILE = "train.feats.csv"


def load_data(file_path):
    df = pd.read_csv(file_path)
    # parse?
    return df


def main():
    data = load_data(path.join(DATA_PATH, TRAIN_FILE))





if __name__ == "__main__":
    main()
