# Adrien Cogny - Assignment 1 for Data Science in the Wild at Cornell Tech
from sklearn.neighbors import NearestNeighbors

class CS5304KNNClassifier():

    numNeighbors = 5;

    classifier = None;
    def __init__(self):
        # Initialization of the knn classifier should be done here
        self.classifier = NearestNeighbors(n_neighbors=self.numNeighbors);        
        return

    def fit(fitX, fitY):
        self.classifier.fit(fitX,fitY);
        return

    def predict(predictX):
        return classifier.predict(predictX)


