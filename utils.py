from typing import Tuple
import numpy as np
import pandas as pd
import math

def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    all_index = list(X.index.values)
    n_samples = X.shape[0]
    train_samples_n = math.ceil(train_proportion * n_samples)
    train_index = all_index[:train_samples_n]
    test_index = list(set(all_index) - set(train_index))

    train_X = X.drop(index=test_index)
    train_y = y.drop(index=test_index)
    test_X = X.drop(index=train_index)
    test_y = y.drop(index=train_index)
    return train_X, train_y, test_X, test_y
