import numpy as np
import utils.mnist_reader as mnist_reader
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

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
    for target in targets:
        indicesToKeep = resultsDF[0] == target
        ax.scatter(resultsDF.loc[indicesToKeep, 'pc0']
                , resultsDF.loc[indicesToKeep, 'pc1']
                , c = colors[target]
                , s = 50)
    ax.legend(targets)
    ax.grid()

    fig.savefig('pca.png')
    print("Saved PCA figure")


applyPCA(X_train,y_train,2)




