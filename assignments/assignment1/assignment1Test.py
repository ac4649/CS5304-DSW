# Adrien Cogny - assignment1Test.py serves as a tester for the classes created
print("Testing Assignment 1");

print("Importing required libraries:");
import numpy as np;
import pandas as pd;

print("Importing the dataset");
from sklearn.datasets import fetch_rcv1
dataSet = fetch_rcv1()

print(dataSet.DESCR);
print(dataSet.data.shape);
print(dataSet.target.shape);
print(dataSet.sample_id.shape);
print(dataSet.target_names.shape);

print("Importing the classes");
from assign1 import CS5304KNNClassifier;

print("Loading the Data into dataframe:");
dataDF = pd.DataFrame(data=dataSet.data,index=dataSet.sample_id);
dataSet.head();


print("---------------------");
print("Testing KNN Classification:");
print("Initializing the classifier");
classifier = CS5304KNNClassifier();

# classifier.fit()


print("---------------------");




print("---------------------");
print("END OF ASSIGNMENT 1 TEST");