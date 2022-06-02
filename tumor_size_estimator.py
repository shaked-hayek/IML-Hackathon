from os import path
import numpy as np
import pandas as pd
import sklearn
from preprocessing import load_data, DATA_PATH, TRAIN_FILE




def main():
    train_X, train_y = load_data(path.join(DATA_PATH, TRAIN_FILE))



if __name__ == "__main__":
    main()

