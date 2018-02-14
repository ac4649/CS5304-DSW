import numpy as np
import pandas as pd
import random
import re
import itertools

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

            train1M[col].value_counts().plot(kind='hist',title=col, bins=100)
            plt.savefig(col)
            plt.show()
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
        
        clear_output()
        print("Extraction Stats: " + str(dataFrame.shape[0]) + " percent: " + str(dataFrame.shape[0] / indexValues.shape[0] * 100) + "%")
        print("Document Stats: " + str(totalNumRowsTraversed) + " percent: " + str(totalNumRowsTraversed/totalNumRows*100) + "%")
        if (dataFrame.shape[0] - prevsize) > 500000:
            prevsize = dataFrame.shape[0]
      
    return dataFrame

# This method generates is a wrapper around the generateSubset to generate the subset and save the dataframe to a csv file (for being able to make use of it after)
def generateAndSaveSubset(file,dataFrame,indexValues,numRowsPerItteration,totalNumRows,column_headers,frameSaveName):
    dataFrame = generateSubSet(file,dataFrame,indexValues,numRowsPerItteration,totalNumRows,column_headers)
    dataFrame.to_csv(frameSaveName)
    return dataFrame

def read_data(data_path, train_path, validation_path, test_path):

    print(data_path)
    print(train_path)
    print(validation_path)
    print(test_path)
    
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
#     print(n)
#     print(data)
    dataFrame = pd.DataFrame(data)
#     print(dataFrame.head())
    dataFrame[dataFrame.columns[0:13]] = dataFrame[dataFrame.columns[0:13]].fillna(0)
    dataFrame[dataFrame.columns[0:13]] = dataFrame[dataFrame.columns[0:13]].replace(-1,0)
    dataFrame[dataFrame.columns[0:13]] = dataFrame[dataFrame.columns[0:13]].replace(-2,0)
#     print(dataFrame[dataFrame.columns[0:13]])
    return np.zeros((data.shape[0], n))


def preprocess_cat_data(data, features, preprocess):
    print(data)
    return None
