import argparse

from assign1 import CS5304KNNClassifier
from assign1 import CS5304NBClassifier
from assign1 import CS5304KMeansClassifier
from assign1 import load_labels, load_training_data, load_validation_data
import pandas as pd
import numpy as np


def load_ks(path_to_ks):
    ks = pd.read_csv(path_to_ks, names=['k'], dtype=np.int32)
    return ks['k'].tolist()


def check_output(output, y):
    assert type(output) == np.ndarray
    assert output.ndim == 1
    assert output.shape[0] == y.shape[0]


if __name__ == '__main__':

    # This represents how the autograder will initialize each model,
    # then call train and predict. The actual autograder will be more
    # involved, since it will calculate the F1 score of the predictions
    # against each label. Note that you are not provided with the test.txt
    # file. It is intentionally hidden.

    # Hint: KMeans' constructor probably will have little functionality.
    # Setting the centroids and any additional training should be done
    # within the train method.

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_labels", default="labels.txt", type=str)
    parser.add_argument("--path_to_ids", default="validation.txt", type=str)
    parser.add_argument("--path_to_ks", default="ks.txt", type=str)
    parser.add_argument("--label", default=33, type=int)
    options = parser.parse_args()

    label = options.label
    ks = load_ks(options.path_to_ks)
    labels = load_labels(options.path_to_labels)
    train_data, train_target, _ = load_training_data()
    eval_data, eval_target, _ = load_validation_data(options.path_to_ids)


    k = ks[labels.index(label)]

    # Grade Ex. 1a
    limit = 1000
    knn = CS5304KNNClassifier(n_neighbors=k)
    knn.train(train_data[:limit], train_target[:limit][:, label])
    output = knn.predict(eval_data[:limit])
    check_output(output,eval_target[:limit])

    # Grade Ex. 1b
    nb = CS5304NBClassifier()
    nb.train(train_data, train_target[:, label])
    output = nb.predict(eval_data)
    check_output(output,eval_target)

    # Grade Ex. 2a
    kmeans = CS5304KMeansClassifier()
    kmeans.train(train_data, train_target[:, label])
    output = kmeans.predict(eval_data)
    check_output(output,eval_target)
