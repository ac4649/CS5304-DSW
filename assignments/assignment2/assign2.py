# template for assign2.py

import numpy as np


def read_data(data_path, train_path, validation_path, test_path):
    train_data, train_target = np.zeros((1000000, 39)), np.zeros((1000000,))
    validation_data, validation_target = np.zeros((250000, 39)), np.zeros((250000,))
    test_data, test_target = np.zeros((750000, 39)), np.zeros((750000,))
    return train_data, train_target, validation_data, validation_target, test_data, test_target


def preprocess_int_data(data, features):
    n = len([f for f in features if f < 13])
    return np.zeros((data.shape[0], n))


def preprocess_cat_data(data, features, preprocess):
    return None
