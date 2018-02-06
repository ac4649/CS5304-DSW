# Adrien Cogny - Assignment 1 for Data Science in the Wild at Cornell Tech
import pandas as pd
import numpy as np

from sklearn.datasets import fetch_rcv1
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.decomposition import TruncatedSVD

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
    def __init__(self,n_neighbors = None):
        if (type(n_neighbors) == int):
            self.numNeighbors = n_neighbors;
        self.classifier = KNeighborsClassifier(n_neighbors=self.numNeighbors,algorithm='brute');        

    def train(self,fitX, fitY):
        #in the fitX, fitY sparce matrix, the index of the element is in indptr

        fitXIndexElem = fitX.indptr;
        fitYIndexElem = fitY.indptr;

        fitXelemDF = pd.Series(data=fitX.indptr[1:])
        fitYelemDF = pd.Series(data=fitY.indptr[1:])
        # print(fitXelemDF);
        # print(fitYelemDF);


        dataArray = fitX.toarray()
        dataDF = pd.DataFrame(data=dataArray,index=fitXelemDF)
        # print(dataDF)

        targetArray = fitY.toarray()
        targetDF = pd.DataFrame(data=targetArray,index=fitYelemDF)
        # print(targetDF)

        self.classifier.fit(dataDF,targetDF);
        return

    def predict(self,predictX):
        dataArray = predictX.toarray()
        dataDF = pd.DataFrame(data=dataArray)
        return self.classifier.predict(predictX)

    def score(self,data,labels):
        dataArray = data.toarray()
        dataDF = pd.DataFrame(data=dataArray)

        predictArray = labels.toarray()
        labelsDF = pd.DataFrame(data=predictArray)

        return self.classifier.score(dataDF,labelsDF)



class CS5304NBClassifier():

    classifier = None;
    def __init__(self, alphanew = 1.25):
        self.classifier = BernoulliNB(alpha = alphanew)
        return
    
    def train(self,fitX, fitY):
        
        self.classifier.fit(fitX,fitY.todense())
        return

    def predict(self,predictX):
        dataArray = predictX.toarray()
        dataDF = pd.DataFrame(data=dataArray)
        return self.classifier.predict(predictX)

    def score(self,data,labels):
        return self.classifier.score(data,labels.todense())

class CS5304KMeansClassifier():
    classifier = None;
#     centroids = None;
    numClusters = 2;
    def __init__(self, n_clusters=2, init='k-means++'):
        self.numClusters = n_clusters
        return
    
    def train(self,fitX, fitY):
        dataset = pd.DataFrame(data = fitX.toarray())
        dataset['label'] = fitY.todense()
        # print(dataset.head(20))
#         dataset.info(memory_usage='deep')
        
#         meanData = dataset[dataset['label'] == 1].drop(columns = ['label']).mean()
#         meanOtherData = dataset[dataset['label'] == 0].drop(columns = ['label']).mean()
        preClassifier = KMeans(n_clusters=1)
        preClassifier.fit(dataset[dataset['label'] == 1].drop(columns = ['label']))
        
        meanCentroid = pd.Series(preClassifier.cluster_centers_[0])
#         print(meanCentroid);
        
        preClassifier.fit(dataset[dataset['label'] == 0].drop(columns = ['label']))
        meanOtherDataCentroid = pd.Series(preClassifier.cluster_centers_[0])
#         print(meanOtherDataCentroid);
        
        
        
        
#         print(meanData)
#         print(meanData.shape)
#         print(meanOtherData)
#         print(meanOtherData.shape)
        
        del dataset # memory drain so delete it
        
        initClusters = pd.concat([meanOtherDataCentroid, meanCentroid], axis = 1);
        
        print(initClusters.head())
        print(initClusters.shape)
        
        
        
        self.classifier = KMeans(n_clusters=self.numClusters, init=initClusters.T)

        self.classifier.fit(fitX,fitY.todense())
        
        return 
    
    def predict(self,predictX):
#         dataArray = predictX.toarray()
#         dataDF = pd.DataFrame(data=dataArray)
        return self.classifier.predict(predictX)

    def score(self,data,labels):
        predictedClusters = self.classifier.predict(data);
        dataFrame = pd.DataFrame(data = predictedClusters, columns=['Predicted'])
        dataFrame['actual'] = labels.todense()
        dataFrame['predictedAccuracy'] = dataFrame['Predicted'] == dataFrame['actual']
        
        numItemsOfClass = (dataFrame['actual'] == 1).sum()
        numItemsNotOfClass = (dataFrame['actual'] == 0).sum()
        
        truePositives = ((dataFrame['actual'] == 1) & (dataFrame['Predicted'] == 1)).sum()
        falsePositives = ((dataFrame['actual'] == 0) & (dataFrame['Predicted'] == 1)).sum()
        falseNegatives = ((dataFrame['actual'] == 1) & (dataFrame['Predicted'] == 0)).sum()
        trueNegatives = ((dataFrame['actual'] == 0) & (dataFrame['Predicted'] == 0)).sum()
        
        Precision = truePositives / (truePositives + falsePositives)
        Recall = truePositives / (truePositives + falseNegatives)
        print(Precision)
        print(Recall)
        f1 = 1/((1/Precision) + (1/Recall))
        return f1