# Adrien Cogny - assignment1Test.py serves as a tester for the classes created
print("Testing Assignment 1");

print("Importing required libraries:");
import numpy as np;
import pandas as pd;
import argparse;

print("Importing the classes");
from assign1 import CS5304KNNClassifier
# from assign1 import CS5304NBClassifier
# from assign1 import CS5304KMeansClassifier
from assign1 import load_labels, load_training_data, load_validation_data

print("Argument Parsing");
parser = argparse.ArgumentParser()
parser.add_argument("--path_to_labels", default="labels.txt", type=str)
parser.add_argument("--path_to_ids", default="validation.txt", type=str)
parser.add_argument("--path_to_ks", default="ks.txt", type=str)
parser.add_argument("--label", default=33, type=int)
options = parser.parse_args()


print("Importing dataSets");
labels = load_labels(options.path_to_labels)
train_data, train_target, _ = load_training_data()
eval_data, eval_target, _ = load_validation_data(options.path_to_ids)

print("Loading the Data into dataframe:");

print("---------------------");
print("Testing KNN Classification:");
print("Initializing the classifier");
classifier = CS5304KNNClassifier();


# classifier.fit()


print("---------------------");




print("---------------------");
print("END OF ASSIGNMENT 1 TEST");