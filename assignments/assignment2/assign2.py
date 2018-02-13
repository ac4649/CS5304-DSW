# template for assign2.py

import numpy as np
import pandas as pd
import random
import re
import itertools


# Additional helper functions required for the code to run:
# This function defines the column headers for the data frames
def getColumnHeaders():
    return pd.Series(data=['Index','label','integer_1','integer_2','integer_3',
                                 'integer_4','integer_5','integer_6','integer_7','integer_8','integer_9'
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

# the generateSubSet function takes in a file, a dataFrame to put the data in as well as index values which should be extracted,
# a number of rows per itteration (this is used to not overload the memory) , total number of rows in the file, the column headers for the new dataframe
def generateSubSet(file,dataFrame,indexValues,numRowsPerItteration,totalNumRows,column_headers):
    totalNumIterations = int(totalNumRows/numRowsPerItteration)
    # print("Number of itterations = " + str(totalNumIterations))
    totalNumRowsTraversed = 0
    prevsize = 0
    for i in range(totalNumIterations + 1):
#         
#         print("Itteration number: " + str(i))
#         print("skipRows: " + str(i * numRowsPerItteration))
#         print("Read in : " + str(numRowsPerItteration))
        curData = pd.read_table(file,skiprows = i * numRowsPerItteration, nrows = numRowsPerItteration,header=None)
        curData.index = [i for i in range(i*numRowsPerItteration,i*numRowsPerItteration + curData.shape[0])]
        totalNumRowsTraversed = totalNumRowsTraversed + curData.shape[0]

#         print(curData.shape)
#         print(curData.index.shape)

        
        curData.columns = column_headers
        curData['Index'] = curData.index
        
        curIndexRange = indexValues['Index'][(indexValues['Index'] < (i*numRowsPerItteration + numRowsPerItteration)) & (indexValues['Index'] > (i*numRowsPerItteration-1))]
        curData = curData[curData['Index'].isin(curIndexRange)]

        dataFrame = pd.concat([dataFrame,curData])
        
        # clear_output()
        # print("Extraction Stats: " + str(dataFrame.shape[0]) + " percent: " + str(dataFrame.shape[0] / indexValues.shape[0] * 100) + "%")
        # print("Document Stats: " + str(totalNumRowsTraversed) + " percent: " + str(totalNumRowsTraversed/totalNumRows*100) + "%")
        if (dataFrame.shape[0] - prevsize) > 500000:
            prevsize = dataFrame.shape[0]
#             dataFrame.to_csv(frameSaveName)
#         elif dataFrame.shape[0] == indexValues.shape[0]:
#             print("Finished with the data collection")
# #             dataFrame.to_csv(frameSaveName)
#             break
    # print("Extraction is Done, now saving frame")        
    return dataFrame

# This method generates is a wrapper around the generateSubset to generate the subset and save the dataframe to a csv file (for being able to make use of it after)
def generateAndSaveSubset(file,dataFrame,indexValues,numRowsPerItteration,totalNumRows,column_headers,frameSaveName):
    dataFrame = generateSubSet(file,dataFrame,indexValues,numRowsPerItteration,totalNumRows,column_headers)
    dataFrame.to_csv(frameSaveName)
    return dataFrame

def read_data(data_path, train_path, validation_path, test_path):

    #Load or Generate the 2M indeces to use
    try:
        twoMIndeces = pd.read_csv('2MIndeces.csv',squeeze = True)
        
    except:
        # print("There were not 2000000 data points")
        twoMIndeces = generateNIndecesFrom(2000000,range(0,45840617)) # this range is because there are this number of records in the training set.
        twoMIndeces.to_csv('2MIndeces.csv',index=False,header=False)


    #Load or Generate the 1M indeces for train, 250k validation and 750k test
    try:
        trainIndeces = pd.read_csv('train_ids.txt',squeeze = True)
        validationIndeces = pd.read_csv('validation_ids.txt',squeeze = True)
    except:
        # print("There were not 1000000 data points")
        trainIndeces = generateNIndecesFrom(1000000,list(twoMIndeces['Index']))
        trainIndeces.to_csv('train_ids.txt',index=False,header=False)

        remainingIndeces = twoMIndeces['Index'][~twoMIndeces['Index'].isin(trainIndeces.values)]
        validationIndeces = generateNIndecesFrom(250000,list(remainingIndeces))
        validationIndeces.to_csv('validation_ids.txt',index=False,header=False)
        
        testingIndeces = twoMIndeces['Index'][~(twoMIndeces['Index'].isin(trainIndeces.values) | twoMIndeces['Index'].isin(validationIndeces.values))]
        testingIndeces = generateNIndecesFrom(750000,list(testingIndeces))
        testingIndeces.to_csv('testing_ids.txt',index=False,header=False)


    # Generate the actual data files
    column_headers = getColumnHeaders()
    try:
        train1M = pd.read_csv('train1M.csv',squeeze = True)
    except:
        # print("No 1M collection")
        train1M = pd.DataFrame()
        train1M = generateAndSaveSubset('dac/train.txt',train1M,trainIndeces,4000000,46000000,column_headers,'train1M.csv')


    try:
        validation250k = pd.read_csv('validation250k.csv',squeeze = True)
    except:
        # print("No 250k collection")
        validation250k = pd.DataFrame()
        validation250k = generateAndSaveSubset('dac/train.txt',validation250k,validationIndeces,4000000,46000000,column_headers,'validation250k.csv')

    try:
        test750k = pd.read_csv('test750k.csv',squeeze = True)
    except:
        # print("No 750k collection")
        test750k = pd.DataFrame()
        test750k = generateAndSaveSubset('dac/train.txt',test750k,validationIndeces,4000000,46000000,column_headers,'test750k.csv')

    # train_data, train_target = np.zeros((1000000, 39)), np.zeros((1000000,))
    # validation_data, validation_target = np.zeros((250000, 39)), np.zeros((250000,))
    # test_data, test_target = np.zeros((750000, 39)), np.zeros((750000,))


    # return train1M.to_array(), trainIndeces.to_array(), validation250k.to_array(), validation_target, test_data, test_target


def preprocess_int_data(data, features):
    n = len([f for f in features if f < 13])
    return np.zeros((data.shape[0], n))


def preprocess_cat_data(data, features, preprocess):
    return None
