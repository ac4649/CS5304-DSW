# Adrien Cogny - assignment1Test.py serves as a tester for the classes created
print("Testing Assignment 1");

print("Importing required libraries:");
import numpy as np;
import pandas as pd;
import argparse;

print("Importing the classes");
from assign1 import CS5304KNNClassifier
from assign1 import CS5304NBClassifier
# from assign1 import CS5304KMeansClassifier
from assign1 import load_labels, load_training_data, load_validation_data


def load_ks(path_to_ks):
    ks = pd.read_csv(path_to_ks, names=['k'], dtype=np.int32)
    return ks['k'].tolist()


def check_output(output, y):
    assert type(output) == np.ndarray
    assert output.ndim == 1
    assert output.shape[0] == y.shape[0]


print("Argument Parsing");
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_labels", default="labels.txt", type=str)
parser.add_argument("--path_to_ids", default="validation.txt", type=str)
parser.add_argument("--path_to_ks", default="ks.txt", type=str)
parser.add_argument("--label", default=33, type=int)
options = parser.parse_args()


print("Importing dataSets");
label = options.label
labels = load_labels(options.path_to_labels)
train_data, train_target, _ = load_training_data()
eval_data, eval_target, _ = load_validation_data(options.path_to_ids)

print("Loading the Data into dataframe:");

print("---------------------");
print("Testing KNN Classification:");
print("Initializing the classifier");

limit = 1000
k = 10 #change this
knn = CS5304KNNClassifier(n_neighbors=k)
print("training and testing classifier");
# print(train_data[:limit])
# print(train_target[:limit][:, label])
knn.train(train_data[:limit], train_target[:limit][:, label])
output = knn.predict(eval_data[:limit])
check_output(output,eval_target[:limit])
print(knn.score(eval_data[:limit],eval_target[:limit][:, label]))

# classifier.fit()
print("---------------------");
print("Testing NB Bernoulli Classification:");
print("Initializing Classifier")
nb = CS5304NBClassifier()

print("training and testing classifier");
nb.train(train_data, train_target[:, label])
output = nb.predict(eval_data)
print(nb.score(eval_data,eval_target[:, label]))
# check_output(output,eval_target)




print("---------------------");
print("END OF ASSIGNMENT 1 TEST");