# Adrien Cogny - Assignment 1 for Data Science in the Wild at Cornell Tech
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_rcv1
from sklearn.neighbors import NearestNeighbors

def load_labels(pathtoLabels):
    DF = pd.read_csv(pathtoLabels,header=None);
    return DF[0];

def load_training_data():
    dataSet = fetch_rcv1(subset='train')

    return dataSet.data, dataSet.target, dataSet.sample_id

def load_validation_data(validationData):

    # This is taken from the homework FAQ and modified slightly.
    test_data = fetch_rcv1(subset='test')
    ids = pd.read_csv(validationData)
    mask = np.isin(test_data.sample_id, ids)
    validation_data = test_data.data[mask]
    validation_target = test_data.target[mask]
    return validation_data, validation_target, test_data

class CS5304KNNClassifier():

    numNeighbors = 5;

    classifier = None;
    # def __init__(self,):
    #     # Initialization of the knn classifier should be done here
    #     self.classifier = NearestNeighbors(n_neighbors=self.numNeighbors);        
    #     return
    def __init__(self,newNumNeighbors = None):
        if (type(newNumNeighbors) == int):
            self.numNeighbors = newNumNeighbors;
        self.classifier = NearestNeighbors(n_neighbors=self.numNeighbors);        

    def train(fitX, fitY):

        targetArray = dataSet['target'].toarray()
        targetDF = pd.DataFrame(data=targetArray,index=dataSet.sample_id,columns=dataSet.target_names)
        targetDF.head()

        self.classifier.fit(fitX,fitY);
        return

    def predict(predictX):
        return classifier.predict(predictX)

    def score(data,labels):
        return classifier.score(data,labels)



class CS5304NBClassifier():

    def __init__(self):
        return

class CS5304KMeansClassifier():

    def __init__(self):
        return