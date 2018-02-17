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

            # these should be commented out when run on jupyter to generate the graphs
            # not tested on regular python:

            # train1M[col][train1M['label'] == 0].value_counts().plot(kind='hist',title=col, bins=100,label='0s')
            # train1M[col][train1M['label'] == 1].value_counts().plot(kind='hist',title=col, bins=100,label='1s')
            # plt.legend(loc='upper right')
            # plt.savefig(col)
#             plt.show()
            # plt.gcf().clear()
            if (train1M[col].dtype != 'O'):
                SummaryStats[col] = train1M[col].describe()   
    # SummaryStats.head()
    SummaryStats.to_csv('integerStats.csv')
    return SummaryStats


# the generateSubSet function takes in a file, a dataFrame to put the data in as well as index values which should be extracted,
# a number of rows per itteration (this is used to not overload the memory) , total number of rows in the file, the column headers for the new dataframe
def generateSubSet(file,dataFrame,indexValues,numRowsPerItteration,totalNumRows,column_headers):
    totalNumIterations = int(totalNumRows/numRowsPerItteration)
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
        
        #get only the top 30 categories with highest count, only 30 because of memory and time limitations.
        # ideally would want to do something like compute the mean count
        #  and take only the values of categories with more than the mean count

        averageNumber = train1M[col].value_counts().mean()
        counts = train1M[col].value_counts()
        topFeatures = train1M[col].value_counts()[:30].index
#         topFeatures = train1M[col].value_counts()[train1M[col].value_counts() > averageNumber].index
        
        categories = pd.Series(topFeatures)
        categories.to_csv(str(col)+'_features.csv',header = False)
        #save the categories for each column
        #then we can set the categegories for each column
        # and when we get dummies from pandas we have a one hot encoding that is consistent accross
        # -> get_dummies() method does one hot encoding

# This methods takes the training set and creates a scaler that is fit to the integer columns of the training set then saves
# The model to file for future retrieval.
def preProcessIntsAndSave(dataFrame,fileName):

    # fill nas and replace negative numbers by 0
    dataFrame[dataFrame.columns[1:14]] = dataFrame[dataFrame.columns[1:14]].fillna(0)
    dataFrame[dataFrame.columns[1:14]] = dataFrame[dataFrame.columns[1:14]].replace(-1,0)
    dataFrame[dataFrame.columns[1:14]] = dataFrame[dataFrame.columns[1:14]].replace(-2,0)
    
    #create the standard scaler
    curScaler = StandardScaler()
    # fit it to the training data (which is the one passed in the dataFrame)
    curScaler.fit(dataFrame[dataFrame.columns[1:14]])
    # save the pickled object to disk for future reference
    joblib.dump(curScaler, fileName)
    return
        
def read_data(data_path, train_path, validation_path, test_path):

    #get the ids
    try:
        trainIndeces = pd.read_csv(train_path, header = None)
        validationIndeces = pd.read_csv(validation_path, header = None)
        testingIndeces = pd.read_csv(test_path, header = None)
    except:
        print("There were not 1000000 data points, generating new points for everything")
        twoMIndeces = generateNIndecesFrom(2000000,range(0,45840617)) # this range is because there are this number of records in the training set.
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
    
    # print("train1M done")
    # Create the summary stats and histograms (the histogram creation works in jupyter notebook but may not work in regular python)
    generateSummaryStatsAndHists(train1M)

    # Select the categories which will be used to one hot encode each data set
    generateCategoricalData(train1M)

    # Create the integers scaler and create the pickled file to load later
    preProcessIntsAndSave(train1M,'scalerPickle.pkl')
    
    validation250k = pd.DataFrame()
    validation250k = generateSubSet(data_path,validation250k,validationIndeces,4000000,46000000,column_headers)

    # print("Validation done")
    
    test750k = pd.DataFrame()
    test750k = generateSubSet(data_path,test750k,testingIndeces,4000000,46000000,column_headers)
    
    # print("test done")
    

    return train1M[train1M.columns[1:40]].values, train1M['label'].values, validation250k[validation250k.columns[1:40]].values, validation250k['label'].values, test750k[test750k.columns[1:40]].values, test750k['label'].values

def preprocess_int_data(data, features):
    n = len([f for f in features if f < 13])
    
    #get the integer data into a dataframe
    dataFrame = pd.DataFrame()
    for f in features:
        if f < 13:
            dataFrame = pd.concat([dataFrame, pd.DataFrame(data[:,f:f+1])],axis=1)

    # get the headers for the dataframe (only the integer ones)
    headers = getDataHeaders()
    trueHeaders = []
    for f in features:
        if f < 13:
            trueHeaders.append(headers[f])
    
    dataFrame.columns = trueHeaders

    # fill nas, and negative numbers with 0s
    for f in features:
        if f < 13:

            dataFrame[dataFrame.columns[f]] = dataFrame[dataFrame.columns[f]].fillna(0)
            dataFrame[dataFrame.columns[f]] = dataFrame[dataFrame.columns[f]].replace(-1,0)
            dataFrame[dataFrame.columns[f]] = dataFrame[dataFrame.columns[f]].replace(-2,0)

    # load the scalar from file and transform the dataFrame (only contains the integers)
    scaler = joblib.load('scalerPickle.pkl') 
    scaledValues = scaler.transform(dataFrame)
    
    return scaledValues


def preprocess_cat_data(data, features, preprocess):
    # create a dataframe with the data
    dataFrame = pd.DataFrame(data)
    
    #create a sparse dataframe that contains the one hot encoded values of each column
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

                # make the dataframe of categorical features actually categorical and not strings
                dataFrame[col] = dataFrame[col].astype('category')

                # load the selected values from the saved file (saved during loading of file)
                curFeatures = pd.read_csv("categorical_" + str(col-12) + "_features.csv",header = None,index_col = 0)

                #set the categories onto the current dataframe
                dataFrame[col].cat.set_categories(curFeatures.values, inplace = True)

                #any non-categorized (not part of the loaded values are going to be dummies)
                dataFrame[col].cat.add_categories(new_categories = 'Dummy',inplace = True)
                dataFrame[col] = dataFrame[col].fillna('Dummy')

                #get the on hot encoding for the column using the values loaded as headers
                onehotVals = pd.get_dummies(dataFrame[col],prefix='encoded_'+ str(col) + "_",sparse=True,columns = curFeatures)

                # concatenate the onhot values for this col with the others.
                returnFrame = pd.concat([returnFrame, onehotVals],axis=1)
    

    return returnFrame.to_coo()