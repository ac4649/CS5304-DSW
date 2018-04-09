import numpy as np
import utils.mnist_reader as mnist_reader
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from fashionmnist import model

import torch

from tqdm import * # remove 


X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = pd.DataFrame(X_train)
y_train = pd.Series(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.Series(y_test)

## applyPCA
def applyPCA(XtrainDF, yTrainSeries, nFeatures):

    XtrainDFStandard = StandardScaler().fit_transform(XtrainDF)
    pca = PCA(n_components=nFeatures)
    XtrainPrincipalComponents = pca.fit_transform(XtrainDFStandard)
    XtrainPrincipalComponentsDF = pd.DataFrame(XtrainPrincipalComponents,columns=['pc'+str(i) for i in range(nFeatures)])

    resultsDF = pd.concat([XtrainPrincipalComponentsDF, yTrainSeries], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = list(set(yTrainSeries.values))
    colors = [
        [1,0,0,1],
        [0.5,0,0,1],
        [0,1,0,1],
        [0,0.5,0,1],
        [0,0,1,1],
        [0,0,0.5,1],
        [1,1,0,1],
        [0.5,0.5,0,1],
        [0,1,1,1],
        [0,0.5,0.5,1]
        ]
    for target in tqdm(targets):
        indicesToKeep = resultsDF[0] == target
        ax.scatter(resultsDF.loc[indicesToKeep, 'pc0']
                , resultsDF.loc[indicesToKeep, 'pc1']
                , c = colors[target]
                , s = 50)
    ax.legend(targets)
    ax.grid()

    fig.savefig('pca.png')
    print("Saved PCA figure")

## ISOMAP

def applyISOMAP(XtrainDF, yTrainSeries, nFeatures):
    
    XtrainDFStandard = StandardScaler().fit_transform(XtrainDF)
    isomap = Isomap(n_components=nFeatures)
    XtrainPrincipalComponents = isomap.fit_transform(XtrainDFStandard)
    XtrainPrincipalComponentsDF = pd.DataFrame(XtrainPrincipalComponents,columns=['pc'+str(i) for i in range(nFeatures)])

    resultsDF = pd.concat([XtrainPrincipalComponentsDF, yTrainSeries], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('ISOMAP Principal Component 1', fontsize = 15)
    ax.set_ylabel('ISOMAP Principal Component 2', fontsize = 15)
    ax.set_title('2 component ISOMAP', fontsize = 20)
    targets = list(set(yTrainSeries.values))
    colors = [
        [1,0,0,1],
        [0.5,0,0,1],
        [0,1,0,1],
        [0,0.5,0,1],
        [0,0,1,1],
        [0,0,0.5,1],
        [1,1,0,1],
        [0.5,0.5,0,1],
        [0,1,1,1],
        [0,0.5,0.5,1]
        ]
    for target in tqdm(targets):
        indicesToKeep = resultsDF[0] == target
        ax.scatter(resultsDF.loc[indicesToKeep, 'pc0']
                , resultsDF.loc[indicesToKeep, 'pc1']
                , c = colors[target]
                , s = 50)
    ax.legend(targets)
    ax.grid()

    fig.savefig('isomap.png')
    print("Saved isomap figure")

    return

def applyTSNE(XtrainDF, yTrainSeries, nFeatures,imageName):
    XtrainDFStandard = StandardScaler().fit_transform(XtrainDF)
    tsnemodel = TSNE(n_components=nFeatures)
    XtrainPrincipalComponents = tsnemodel.fit_transform(XtrainDFStandard)
    XtrainPrincipalComponentsDF = pd.DataFrame(XtrainPrincipalComponents,columns=['pc'+str(i) for i in range(nFeatures)])

    resultsDF = pd.concat([XtrainPrincipalComponentsDF, yTrainSeries], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('TSNE Principal Component 1', fontsize = 15)
    ax.set_ylabel('TSNE Principal Component 2', fontsize = 15)
    ax.set_title('2 component TSNE', fontsize = 20)
    targets = list(set(yTrainSeries.values))
    colors = [
        [1,0,0,1],
        [0.5,0,0,1],
        [0,1,0,1],
        [0,0.5,0,1],
        [0,0,1,1],
        [0,0,0.5,1],
        [1,1,0,1],
        [0.5,0.5,0,1],
        [0,1,1,1],
        [0,0.5,0.5,1]
        ]
    for target in tqdm(targets):
        indicesToKeep = resultsDF[0] == target
        ax.scatter(resultsDF.loc[indicesToKeep, 'pc0']
                , resultsDF.loc[indicesToKeep, 'pc1']
                , c = colors[target]
                , s = 50)
    ax.legend(targets)
    ax.grid()

    fig.savefig(imageName)
    print("Saved isomap figure")

    return


# applyPCA(X_train,y_train,2)

#training isomap on 30% of the data
subsetX_train = X_train.sample(frac=0.30)
subsetY_train = y_train.iloc[subsetX_train.index.values]
print("Sample Size: " + str(subsetX_train.shape[0]))
# applyISOMAP(subsetX_train,subsetY_train,2)


applyTSNE(subsetX_train,subsetY_train,2,'tsne-raw.png')

# using the trained model
# load the model
model = torch.load('/fashion-mnist/FashionSimpleNet-run-1.pth.tar')
print(model)
# applyTSNE(subsetX_train,subsetY_train,2,'tsne-resnet.png')




