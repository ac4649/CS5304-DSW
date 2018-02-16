import numpy as np
import pandas as pd
import random
import re
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib 


# Additional helper functions required for the code to run:
# This function defines the column headers for the data frames
def getColumnHeaders():
    return pd.Series(data=['label','integer_1','integer_2','integer_3',
                                 'integer_4','integer_5','integer_6','integer_7','integer_8','integer_9',
                                 'integer_10','integer_11','integer_12','integer_13','categorical_1',
                                 'categorical_2','categorical_3','categorical_4','categorical_5','categorical_6',
                                 'categorical_7','categorical_8','categorical_9','categorical_10','categorical_11',
                                 'categorical_12','categorical_13','categorical_14','categorical_15','categorical_16',
                                 'categorical_17','categorical_18','categorical_19','categorical_20','categorical_21',
                                 'categorical_22','categorical_23','categorical_24','categorical_25','categorical_26','Index'])

def getDataHeaders():
    return pd.Series(data=['integer_1','integer_2','integer_3',
                                 'integer_4','integer_5','integer_6','integer_7','integer_8','integer_9',
                                 'integer_10','integer_11','integer_12','integer_13','categorical_1',
                                 'categorical_2','categorical_3','categorical_4','categorical_5','categorical_6',
                                 'categorical_7','categorical_8','categorical_9','categorical_10','categorical_11',
                                 'categorical_12','categorical_13','categorical_14','categorical_15','categorical_16',
                                 'categorical_17','categorical_18','categorical_19','categorical_20','categorical_21',
                                 'categorical_22','categorical_23','categorical_24','categorical_25','categorical_26'])

# this function generateNIndecesFrom takes a range of Indeces and randomly takes n of them, returns a pandas Series object
def generateNIndecesFrom(n, rangeOfIndeces):
    # print("Generating " + str(n) + " indeces from range")
    allIndeces = random.sample(rangeOfIndeces, n)
    allIndeces = pd.Series(data = allIndeces)
    allIndeces = allIndeces.sort_values().reset_index().drop(['index'],axis=1)
    allIndeces.columns = ['Index'];
    return allIndeces

# This function takes in a dataframe and computes statistics and histograms for all columns that are not 'label', 'Index' or 'Unnamed: 0'
# It was used in part 2.2 to generate the histograms and statistics.
# It can for example be placed at the end of the read_data method and be passed one of the datasets to compute its statistics and histograms
def generateSummaryStatsAndHists(train1M):
    SummaryStats = pd.DataFrame()
    for col in train1M.columns:
        if (col != 'label' and col != 'Index' and col != 'Unnamed: 0'):

            train1M[col][train1M['label'] == 0].value_counts().plot(kind='hist',title=col, bins=100,label='0s')
            train1M[col][train1M['label'] == 1].value_counts().plot(kind='hist',title=col, bins=100,label='1s')
            plt.legend(loc='upper right')
            plt.savefig(col)
#             plt.show()
            plt.gcf().clear()
            if (train1M[col].dtype != 'O'):
                SummaryStats[col] = train1M[col].describe()   
    # SummaryStats.head()
    SummaryStats.to_csv('integerStats.csv')
    return SummaryStats


# the generateSubSet function takes in a file, a dataFrame to put the data in as well as index values which should be extracted,
# a number of rows per itteration (this is used to not overload the memory) , total number of rows in the file, the column headers for the new dataframe
def generateSubSet(file,dataFrame,indexValues,numRowsPerItteration,totalNumRows,column_headers):
    totalNumIterations = int(totalNumRows/numRowsPerItteration)
    # print("Number of itterations = " + str(totalNumIterations))
    totalNumRowsTraversed = 0
    prevsize = 0
    for i in range(totalNumIterations + 1):

        curData = pd.read_table(file,skiprows = i * numRowsPerItteration, nrows = numRowsPerItteration,header=None)
        curData.index = [i for i in range(i*numRowsPerItteration,i*numRowsPerItteration + curData.shape[0])]
        totalNumRowsTraversed = totalNumRowsTraversed + curData.shape[0]
        

        curData['Index'] = curData.index
        curData.columns = column_headers
        
        curIndexRange = indexValues['Index'][(indexValues['Index'] < (i*numRowsPerItteration + numRowsPerItteration)) & (indexValues['Index'] > (i*numRowsPerItteration-1))]
        curData = curData[curData['Index'].isin(list(curIndexRange))]
        
        dataFrame = pd.concat([dataFrame,curData])
        
#         clear_output()
#         print("Extraction Stats: " + str(dataFrame.shape[0]) + " percent: " + str(dataFrame.shape[0] / indexValues.shape[0] * 100) + "%")
#         print("Document Stats: " + str(totalNumRowsTraversed) + " percent: " + str(totalNumRowsTraversed/totalNumRows*100) + "%")
        if (dataFrame.shape[0] - prevsize) > 500000:
            prevsize = dataFrame.shape[0]
      
    return dataFrame

# This method generates is a wrapper around the generateSubset to generate the subset and save the dataframe to a csv file (for being able to make use of it after)
def generateAndSaveSubset(file,dataFrame,indexValues,numRowsPerItteration,totalNumRows,column_headers,frameSaveName):
    dataFrame = generateSubSet(file,dataFrame,indexValues,numRowsPerItteration,totalNumRows,column_headers)
    dataFrame.to_csv(frameSaveName)
    return dataFrame

# This method generates the categorical data required to then apply one hot encoding on the entire dataset
def generateCategoricalData(train1M):
    #change to categorical
    for col in train1M.columns[14:40]:
        train1M[col] = train1M[col].astype('category')
        
        #get only the top 10 categories with highest count
        averageNumber = train1M[col].value_counts().mean()
#         print(averageNumber)
        counts = train1M[col].value_counts()
# #         print(counts)
#         if (averageNumber < 10):
#             # if the average counte is less than 10, just take the first 20 categories
#             topFeatures = train1M[col].value_counts()[:20].index
#         else:
        topFeatures = train1M[col].value_counts()[train1M[col].value_counts() > averageNumber].index

#         print(topFeatures)
        
        # add the dummy category
#         train1M[col].cat.add_categories(new_categories = 'Dummy',inplace = True)
        categories = pd.Series(topFeatures)
#         print(categories.shape)
        categories.to_csv(str(col)+'_features.csv',header = False)
        #save the categories for each column
        #then we can set the categegories for each column
        # and when we get dummies from pandas we have a one hot encoding that is consistent accross
        # -> get_dummies() method does one hot encoding

# This methods takes the training set and creates a scaler that is fit to the integer columns of the training set then saves
# The model to file for future retrieval.
def preProcessIntsAndSave(dataFrame,fileName):
    dataFrame[dataFrame.columns[1:14]] = dataFrame[dataFrame.columns[1:14]].fillna(0)
    dataFrame[dataFrame.columns[1:14]] = dataFrame[dataFrame.columns[1:14]].replace(-1,0)
    dataFrame[dataFrame.columns[1:14]] = dataFrame[dataFrame.columns[1:14]].replace(-2,0)
    
    curScaler = StandardScaler()
    curScaler.fit(dataFrame[dataFrame.columns[1:14]])
    joblib.dump(curScaler, fileName)
    return
        
def read_data(data_path, train_path, validation_path, test_path):

#     print(data_path)
#     print(train_path)
#     print(validation_path)
#     print(test_path)
    
    #get the ids
    try:
        trainIndeces = pd.read_csv(train_path, header = None)
        validationIndeces = pd.read_csv(validation_path, header = None)
        testingIndeces = pd.read_csv(test_path, header = None)
    except:
        print("There were not 1000000 data points")
        trainIndeces = generateNIndecesFrom(1000000,list(twoMIndeces['Index']))
        trainIndeces.to_csv('train_ids.txt',index=False,header=False)

        remainingIndeces = twoMIndeces['Index'][~twoMIndeces['Index'].isin(trainIndeces.values)]
        validationIndeces = generateNIndecesFrom(250000,list(remainingIndeces))
        validationIndeces.to_csv('validation_ids.txt',index=False,header=False)

        testingIndeces = twoMIndeces['Index'][~(twoMIndeces['Index'].isin(trainIndeces.values) | twoMIndeces['Index'].isin(validationIndeces.values))]
        testingIndeces = generateNIndecesFrom(750000,list(testingIndeces))
        testingIndeces.to_csv('test_ids.txt',index=False,header=False)
    
    trainIndeces.columns = ['Index']
    validationIndeces.columns = ['Index']
    testingIndeces.columns = ['Index']

    # Generate the actual data files
    column_headers = getColumnHeaders()
    train1M = pd.DataFrame()
    train1M = generateSubSet(data_path,train1M,trainIndeces,4000000,46000000,column_headers)
#     return train1M
    generateSummaryStatsAndHists(train1M)
    generateCategoricalData(train1M)
    preProcessIntsAndSave(train1M,'scalerPickle.pkl')
    
    validation250k = pd.DataFrame()
    validation250k = generateSubSet(data_path,validation250k,validationIndeces,4000000,46000000,column_headers)

    test750k = pd.DataFrame()
    test750k = generateSubSet(data_path,test750k,testingIndeces,4000000,46000000,column_headers)
    
    
#     print(train1M.shape)
#     print(validation250k.shape)
#     print(test750k.shape)

    return train1M[train1M.columns[1:40]].values, train1M['label'].values, validation250k[validation250k.columns[1:40]].values, validation250k['label'].values, test750k[test750k.columns[1:40]].values, test750k['label'].values

def preprocess_int_data(data, features):
    n = len([f for f in features if f < 13])
    
    dataFrame = pd.DataFrame()
    for f in features:
        if f < 13:
            dataFrame = pd.concat([dataFrame, pd.DataFrame(data[:,f:f+1])],axis=1)

    headers = getDataHeaders()
    trueHeaders = []
    for f in features:
        if f < 13:
            trueHeaders.append(headers[f])
    
    dataFrame.columns = trueHeaders

    for f in features:
        if f < 13:

            dataFrame[dataFrame.columns[f]] = dataFrame[dataFrame.columns[f]].fillna(0)
            dataFrame[dataFrame.columns[f]] = dataFrame[dataFrame.columns[f]].replace(-1,0)
            dataFrame[dataFrame.columns[f]] = dataFrame[dataFrame.columns[f]].replace(-2,0)

    scaler = joblib.load('scalerPickle.pkl') 
    scaledValues = scaler.transform(dataFrame)
    
    return scaledValues


def preprocess_cat_data(data, features, preprocess):
#     print(features)
#     print(preprocess)
#     print(data)
    dataFrame = pd.DataFrame(data)
#     print(dataFrame[dataFrame.columns[13:39]])
    # Change each column in the 13-39 into categorical
#     dataFrame.columns = getDataHeaders()
    
    returnFrame = pd.SparseDataFrame()
    # drop the cols that are not in the features vector
    for col in dataFrame.columns:
        foundCol = False
        for f in features:
            if (f == col):
                foundCol = True
                
        if foundCol == False:
            dataFrame.drop(col,inplace=True,axis=1)
        else:    
            # I know that the categorical features start at 1 and index 13 so add 12 to f
            if (col > 12):
                print(col)
#                 print(dataFrame[col].dtype)
                dataFrame[col] = dataFrame[col].astype('category')
                curFeatures = pd.read_csv("categorical_" + str(col-12) + "_features.csv",header = None,index_col = 0)
#                 print(curFeatures.values)
                dataFrame[col].cat.set_categories(curFeatures.values)
                dataFrame[col].cat.add_categories(new_categories = 'Dummy',inplace = True)
                dataFrame[col] = dataFrame[col].fillna('Dummy')
#                 print(dataFrame[col].dtype)
# #                 print(dataFrame[col].cat.categories)
                onehotVals = pd.get_dummies(dataFrame[col],prefix='encoded_'+ str(col) + "_",sparse=True)
                print(onehotVals.info())
                returnFrame = pd.concat([returnFrame, onehotVals],axis=1)
                print("Got 1hot for " + str(col))
#                 print(returnFrame.info())
#                 return
    
#     print(dataFrame.head())
#     print(dataFrame.info())
#     print(returnFrame.head())
    
#     for col in dataFrame.columns[13:39]:
#         dataFrame[col] = dataFrame[col].astype('category')
#         #reset the categories to the ones for that column
#         
        
#         dataFrame[col].cat.set_categories(curFeatures.values)
#         pd.get_dummies(train1M[col],prefix=['encoded'],sparse=True)
#     dataFrame[dataFrame.columns[13:39]] = dataFrame[dataFrame.columns[13:39]].fillna('Dummy')
    return returnFrame.values