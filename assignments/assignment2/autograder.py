# sample autograder for assign2.py

import argparse

from assign2 import read_data
from assign2 import preprocess_int_data
from assign2 import preprocess_cat_data


def check_output_features(fs):
    assert len(fs) < 39


def check_output_read_data(data, target, n):
    assert data.shape[0] == n
    assert target.shape[0] == n


def check_output_preprocess(preprocess):
    assert (preprocess == 'onehot') or (preprocess == 'rate') or (preprocess == 'tfidf')


def check_output_preprocess_int_data(data, fs):
    n = len([f for f in fs if f < 13])
    assert data.shape[1] == n


def check_output_preprocess_cat_data(data, fs, preprocess):
    pass


def read_features(path):
    features = []
    with open(path) as f:
        for line in f:
            features.append(int(line.strip()))
    return features


def read_preprocess(path):
    with open(path) as f:
        for line in f:
            return line.strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./dac/train.txt', type=str)
    parser.add_argument('--train', default='train_ids.txt', type=str)
    parser.add_argument('--validation', default='validation_ids.txt', type=str)
    parser.add_argument('--test', default='test_ids.txt', type=str)
    parser.add_argument('--features', default='features.txt', type=str)
    parser.add_argument('--preprocess', default='preprocess.txt', type=str)
    options = parser.parse_args()

    train_data, train_target, validation_data, validation_target, test_data, test_target = \
        read_data(options.data, options.train, options.validation, options.test)

    check_output_read_data(train_data, train_target, 1000000)
    check_output_read_data(validation_data, validation_target, 250000)
    check_output_read_data(test_data, test_target, 750000)

    features = read_features(options.features)

    check_output_features(features)

    preprocess = read_preprocess(options.preprocess)

    check_output_preprocess(preprocess)

    train_int_data = preprocess_int_data(train_data, features)
    validation_int_data = preprocess_int_data(validation_data, features)
    test_int_data = preprocess_int_data(test_data, features)

    check_output_preprocess_int_data(train_int_data, features)
    check_output_preprocess_int_data(validation_int_data, features)
    check_output_preprocess_int_data(test_int_data, features)

    train_cat_data = preprocess_cat_data(train_data, features, preprocess)
    validation_cat_data = preprocess_cat_data(validation_data, features, preprocess)
    test_cat_data = preprocess_cat_data(test_data, features, preprocess)

    check_output_preprocess_cat_data(train_cat_data, features, preprocess)
    check_output_preprocess_cat_data(validation_cat_data, features, preprocess)
    check_output_preprocess_cat_data(test_cat_data, features, preprocess)
