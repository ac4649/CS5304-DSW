import numpy as np
from scipy.sparse import rand as sprand
from scipy.sparse import lil_matrix
import torch
from torch.autograd import Variable
import pandas as pd


class MatrixFactorization(torch.nn.Module):


    def __init__(self, n_users, n_items, n_factors=5, useBias = False):
        super().__init__()
        
        if torch.cuda.is_available(): # CHECK FOR CUDA AVAILABILITY
            self.useCUDA = True
            print("CUDA is being used")
        else:
            print("CUDA not available, reverting to CPU")
            
        if self.useCUDA: # IF IT IS AVAILABLE, USE IT
            self.cuda() 
        
        self.user_factors = torch.nn.Embedding(n_users, 
                                               n_factors,
                                               sparse=False)
        self.item_factors = torch.nn.Embedding(n_items, 
                                               n_factors,
                                               sparse=False)
        # Also should consider fitting overall bias (self.mu term) and both user and item bias vectors
        
        ## Incorporation of Bias Term
        if useBias:
            self.useBias = True
            self.user_bias = torch.nn.Embedding(n_users, 1, sparse = False)
            self.item_bias = torch.nn.Embedding(n_items, 1, sparse = False)
        
        
        # Mu is 1x1, user_bias is 1xn_users. item_bias is 1xn_items
        
    def getLoss(self):
        return self.currentLoss.data.cpu().numpy()[0]


    currentLoss = 10
    useCUDA = False
    useBias = False
    
    interactions = False
#     loss_func = torch.nn.MSELoss()
#     reg_loss_func = torch.optim.SGD(self.model.parameters(), lr=1e-6, weight_decay=1e-3)
    
    # For convenience when we want to predict a single user-item pair. 
    def predict(self, user, item):
        # Need to fit bias factorsx
        print("Predicting")
        
        print(self.user_factors(user))
        print(torch.transpose(self.item_factors(item),0,1))

#         return torch.mm(self.user_factors(user)[0],torch.transpose(self.item_factors(item),0,1)[0])
        # return torch.mm(self.user_factors(user),torch.transpose(self.item_factors(item),0,1))
        if (self.useBias):
            print(self.user_bias(user))
            print(self.item_bias(item))
            pred = self.user_bias(user) + self.item_bias(item)
            pred += (self.user_factors(user) * self.item_factors(item)).sum(1)
            return pred
        else:
            return torch.mm(self.user_factors(user),torch.transpose(self.item_factors(item),0,1))
    
#         return torch.dot(self.user_factors(user),self.item_factors(item))
    
    # Much more efficient batch operator. This should be used for training purposes
    def forward(self, users, items):
        # Need to fit bias factors
#         print("Forward")
        if (self.useBias):
            # print(self.user_bias(users))
            # print(self.item_bias(items))
            # pred = self.user_bias(users) + self.item_bias(items)
            # pred += (self.user_factors(users) * self.item_factors(items)).sum(1)
            # return pred
            pred = torch.mm(self.user_factors(users),torch.transpose(self.item_factors(items),0,1))
            # pred += user_bias.expand_as(pred)
            pred += torch.transpose(self.item_bias(items),0,1).expand_as(pred)
            return pred
            # return torch.mm(self.user_bias(users) + self.user_factors(users),torch.transpose(self.item_bias(items) + self.item_factors(items),0,1))
        else:
            return torch.mm(self.user_factors(users),torch.transpose(self.item_factors(items),0,1))
    
    def get_batch(self,batch_size,ratings):
        # Sort our data and scramble it
        rows, cols = ratings.shape
        p = np.random.permutation(rows)

        # create batches
        sindex = 0
        eindex = batch_size
        while eindex < rows:
            batch = p[sindex:eindex]
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield batch

        if eindex >= rows:
            batch = range(sindex,rows)
            yield batch 
    
    def run_test(self,batch_size,ratings_test):

        predictionsArray = None # start the predictions array as -1 then in the loop create the np array returned
        losses = []
        print("Number Batches: {}".format(ratings_test.shape[0]/batch_size))
        for i,batch in enumerate(self.get_batch(batch_size,ratings_test)):
            # print(i)
            # print(batch)
            if self.useCUDA:
                interactions = Variable(torch.cuda.FloatTensor(ratings_test[batch, :].toarray()),volatile = True)
                rows = Variable(torch.cuda.LongTensor(batch), volatile = True)
                cols = Variable(torch.cuda.LongTensor(np.arange(ratings_test.shape[1])), volatile = True)               
            else:
                interactions = Variable(torch.FloatTensor(ratings_test[batch, :].toarray()),volatile = True)
                rows = Variable(torch.LongTensor(batch),volatile = True)
                cols = Variable(torch.LongTensor(np.arange(ratings_test.shape[1])),volatile = True)

            # Predict and calculate loss
            predictions = self(rows, cols)
            # predictions = self.predict(rows,cols)

            if type(predictionsArray) is not np.ndarray:
                predictionsArray = predictions.data.cpu().numpy()
            else:
                predictionsArray = np.append(predictionsArray,predictions.data.cpu().numpy(),axis = 0)
            # print(predictionsArray.shape)
            # print(type(self.loss_func(predictions, interactions)))
            losses.append(self.loss_func(predictions, interactions).data.cpu().numpy())

            # del interactions
            # del rows
            # del cols
        return predictionsArray, losses
    
    def run_epoch(self,batch_size, ratings):
        for i, batch in enumerate(self.get_batch(batch_size, ratings)):
            # Set gradients to zero
            self.reg_loss_func.zero_grad()

#             print(type(batch))
            # Turn data into variables
            if self.useCUDA:
#                 print("using cuda")
                interactions = Variable(torch.cuda.FloatTensor(ratings[batch, :].toarray()))
                rows = Variable(torch.cuda.LongTensor(batch))
                cols = Variable(torch.cuda.LongTensor(np.arange(ratings.shape[1])))               
            else:
                interactions = Variable(torch.FloatTensor(ratings[batch, :].toarray()))
                rows = Variable(torch.LongTensor(batch))
                cols = Variable(torch.LongTensor(np.arange(ratings.shape[1])))

            predictions = self(rows, cols)
            self.currentLoss = self.loss_func(predictions, interactions)

            # Backpropagate
            self.currentLoss.backward()

            # Update the parameters
            self.reg_loss_func.step()
    
    def train(self, numEpochs, batch_size, ratings,learningRate):
        self.loss_func = torch.nn.MSELoss()
        self.reg_loss_func = torch.optim.SGD(self.parameters(), lr=learningRate, weight_decay=1e-3)
        for i in range(numEpochs):
            print(i)
            self.run_epoch(batch_size,ratings)
            
    def convertArryToVariable(self, array):
        if self.useCUDA:
            return Variable(torch.cuda.LongTensor(array))
        else:
            return Variable(torch.LongTensor(array))


    
    def convertLillMatrixToVariable(self,lillMatrix):
        if self.useCUDA:
            if lillMatrix.shape[0] == 1:
                # we have a single matrix
                return Variable(torch.cuda.LongTensor(lillMatrix.toarray()))
            else:
                return Variable(torch.cuda.LongTensor(lillMatrix.toarray()))
        else:
            if lillMatrix.shape[0] == 1:
                # we have a single matrix
                return Variable(torch.LongTensor(lillMatrix.toarray()))
            else:
                return Variable(torch.LongTensor(lillMatrix.toarray()))


# This function gets the size of users 
def getUserItemNumbers(trainDF,testDF):
    n_userTrain = max(trainDF.user_id.unique())
    n_itemsTrain = max(trainDF.item_id.unique())
    
    n_userTest = max(testDF.user_id.unique())
    n_itemsTest = max(testDF.item_id.unique())

    return max(n_userTrain,n_userTest), max(n_itemsTrain,n_itemsTest)


def get_movielens_ratings(df):
    n_users = max(df.user_id.unique())
    n_items = max(df.item_id.unique())
    # print(n_users)
    # print(n_items)
    interactions = lil_matrix( (n_users,n_items), dtype=float) #np.zeros((n_users, n_items))
    for row in df.itertuples():
        interactions[row[1] - 1, row[2] - 1] = row[3]
    return interactions

def get_movielens_ratings_testLarger(df,n_users,n_items):

    start_user = min(df.user_id.unique())
    start_item = 0
    n_users = n_users - start_user
    # print(start_user)
    # print(n_users)
    n_items = n_items - start_item
    # print(start_item)
    # print(n_items)
    interactions = lil_matrix( (n_users,n_items), dtype=float) #np.zeros((n_users, n_items))
    for row in df.itertuples():
        # print(row[1])
        # print(row[2])
        # print(row[3])
        interactions[row[1] - 1 - start_user, row[2] - 1 - start_item] = row[3]
    return interactions


def RunTask1():
    print("Running Task 1:")

    loadPrevResults = False
    loadSavedMeans = False

    EPOCH = 5 # Number of Epochs to train for
    BATCH_SIZE = 1000 #50
    LRs = [0.001, 0.01, 0.1] # array of learning rates to test

    FileNames = [
        "r1"
        ,
        "r2"
        ,
        "r3"
        ,
        "r4"
        ,
        "r5"
        ]
    if loadPrevResults:
        LambdaMeanLossResultsFrame = pd.read_csv('Task1_LambdaMeanLossResults_{}Epochs.csv'.format(EPOCH),index_col=0)

        print(LambdaMeanLossResultsFrame)
    else:
        LambdaMeanLossResultsFrame = pd.DataFrame(index=["r1","r2","r3","r4","r5"], columns=['0.001','0.01','0.1'])
        print(LambdaMeanLossResultsFrame)



    print("Running Exmperiment on file ids {}".format(FileNames))
    for fileName in FileNames:

        print("Starting Building model for cross validation {}".format(fileName))
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        print("Loading Data")
        df_train = pd.read_csv('ml-10M100K/'+fileName+'.train', sep='::', names=names,engine='python')
        print("Loaded train data")
        df_test = pd.read_csv('ml-10M100K/'+fileName+'.test', sep='::', names=names,engine='python')
        print("Loaded Data")


        if fileName == 'r5': # This is the special case where the test set is at the end of the file so the ids are larger.
            numUsers, numItems = getUserItemNumbers(df_train,df_test)

            ratings = get_movielens_ratings(df_train)
            # ratings = get_movielens_ratings(df_train)
            print(ratings.shape)
            test_ratings = get_movielens_ratings_testLarger(df_test,numUsers,numItems)
            # test_ratings = get_movielens_ratings(df_test)
            print(test_ratings.shape)
        else:
            # ratings = get_movielens_ratings(df_train,numUsers,numItems)
            ratings = get_movielens_ratings(df_train)
            print(ratings.shape)
            # test_ratings = get_movielens_ratings(df_test,numUsers,numItems)
            test_ratings = get_movielens_ratings(df_test)
            print(test_ratings.shape)

        print("Prepared the data")


        print("Creating Model")
        model = MatrixFactorization(ratings.shape[0], ratings.shape[1], n_factors=2,useBias = False)
        if torch.cuda.is_available():
            model.cuda()

        print("Running the training")

        print("Learining Rates tested: {}".format(LRs))
        for LR in LRs:
            print("Training Model")
            print("Number iterations Per Epoch: {}".format(ratings.shape[0]/BATCH_SIZE))
            model.train(EPOCH,BATCH_SIZE,ratings,LR)

            print("Model Loss: {}".format(model.getLoss()))
            print("Running Test")
            predictions, losses = model.run_test(100,test_ratings)

            print(len(predictions))
            print(np.mean(losses))

            LambdaMeanLossResultsFrame.loc[fileName][str(LR)] = np.mean(losses)
            print(LambdaMeanLossResultsFrame)

            LambdaMeanLossResultsFrame.to_csv('Task1_LambdaMeanLossResults_{}Epochs.csv'.format(EPOCH))

        LambdaMeanLossResultsFrame.to_csv('Task1_LambdaMeanLossResults_{}Epochs.csv'.format(EPOCH))

    LambdaMeanLossResultsFrame.to_csv('Task1_LambdaMeanLossResults_{}Epochs.csv'.format(EPOCH))

    print(LambdaMeanLossResultsFrame)

def RunTask2():
    print("Running Task 2:")

    loadPrevResults = False
    loadSavedMeans = False

    EPOCH = 5 # Number of Epochs to train for
    BATCH_SIZE = 1000 #50
    LRs = [0.001, 0.01, 0.1] # array of learning rates to test

    FileNames = [
        "r1"
        ,
        "r2"
        ,
        "r3"
        ,
        "r4"
        ,
        "r5"
        ]
    if loadPrevResults:
        LambdaMeanLossResultsFrame = pd.read_csv('Task2_LambdaMeanLossResults_{}Epochs.csv'.format(EPOCH),index_col=0)

        print(LambdaMeanLossResultsFrame)
    else:
        LambdaMeanLossResultsFrame = pd.DataFrame(index=["r1","r2","r3","r4","r5"], columns=['0.001','0.01','0.1'])
        print(LambdaMeanLossResultsFrame)



    print("Running Exmperiment on file ids {}".format(FileNames))
    for fileName in FileNames:

        print("Starting Building model for cross validation {}".format(fileName))
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        print("Loading Data")
        df_train = pd.read_csv('ml-10M100K/'+fileName+'.train', sep='::', names=names,engine='python')
        print("Loaded train data")
        df_test = pd.read_csv('ml-10M100K/'+fileName+'.test', sep='::', names=names,engine='python')
        print("Loaded Data")


        if fileName == 'r5': # This is the special case where the test set is at the end of the file so the ids are larger.
            numUsers, numItems = getUserItemNumbers(df_train,df_test)

            ratings = get_movielens_ratings(df_train)
            # ratings = get_movielens_ratings(df_train)
            print(ratings.shape)
            test_ratings = get_movielens_ratings_testLarger(df_test,numUsers,numItems)
            # test_ratings = get_movielens_ratings(df_test)
            print(test_ratings.shape)
        else:
            # ratings = get_movielens_ratings(df_train,numUsers,numItems)
            ratings = get_movielens_ratings(df_train)
            print(ratings.shape)
            # test_ratings = get_movielens_ratings(df_test,numUsers,numItems)
            test_ratings = get_movielens_ratings(df_test)
            print(test_ratings.shape)

        print("Prepared the data")


        print("Creating Model")
        model = MatrixFactorization(ratings.shape[0], ratings.shape[1], n_factors=2,useBias = True)
        if torch.cuda.is_available():
            model.cuda()

        print("Running the training")

        print("Learining Rates tested: {}".format(LRs))
        for LR in LRs:
            print("Training Model")
            print("Number iterations Per Epoch: {}".format(ratings.shape[0]/BATCH_SIZE))
            model.train(EPOCH,BATCH_SIZE,ratings,LR)

            print("Model Loss: {}".format(model.getLoss()))
            print("Running Test")
            predictions, losses = model.run_test(100,test_ratings)

            print(len(predictions))
            print(np.mean(losses))

            LambdaMeanLossResultsFrame.loc[fileName][str(LR)] = np.mean(losses)
            print(LambdaMeanLossResultsFrame)

            LambdaMeanLossResultsFrame.to_csv('Task2_LambdaMeanLossResults_{}Epochs.csv'.format(EPOCH))

        LambdaMeanLossResultsFrame.to_csv('Task2_LambdaMeanLossResults_{}Epochs.csv'.format(EPOCH))

    LambdaMeanLossResultsFrame.to_csv('Task2_LambdaMeanLossResults_{}Epochs.csv'.format(EPOCH))

    print(LambdaMeanLossResultsFrame)


def RunTask3():
    print("Running Task 3:")

    EPOCH = 1 # Number of Epochs to train for
    BATCH_SIZE = 1000 #50
    LRs = [0.1] # array of learning rates to test

    FileNames = ["r5"]

    print("Running Exmperiment on file ids {}".format(FileNames))
    for fileName in FileNames:

        print("Starting Building model for cross validation {}".format(fileName))
        names = ['user_id', 'item_id', 'rating', 'timestamp']
        print("Loading Data")
        df_train = pd.read_csv('ml-10M100K/'+fileName+'.train', sep='::', names=names,engine='python')
        print("Loaded train data")
        df_test = pd.read_csv('ml-10M100K/'+fileName+'.test', sep='::', names=names,engine='python')
        print("Loaded Data")


        if fileName == 'r5': # This is the special case where the test set is at the end of the file so the ids are larger.
            numUsers, numItems = getUserItemNumbers(df_train,df_test)

            ratings = get_movielens_ratings(df_train)
            # ratings = get_movielens_ratings(df_train)
            print(ratings.shape)
            test_ratings = get_movielens_ratings_testLarger(df_test,numUsers,numItems)
            # test_ratings = get_movielens_ratings(df_test)
            print(test_ratings.shape)
        else:
            # ratings = get_movielens_ratings(df_train,numUsers,numItems)
            ratings = get_movielens_ratings(df_train)
            print(ratings.shape)
            # test_ratings = get_movielens_ratings(df_test,numUsers,numItems)
            test_ratings = get_movielens_ratings(df_test)
            print(test_ratings.shape)

        print("Prepared the data")


        print("Creating Model")
        model = MatrixFactorization(ratings.shape[0], ratings.shape[1], n_factors=2,useBias = False)
        if torch.cuda.is_available():
            model.cuda()

        print("Running the training")

        print("Learining Rates tested: {}".format(LRs))
        for LR in LRs:
            print("Training Model")
            print("Number iterations Per Epoch: {}".format(ratings.shape[0]/BATCH_SIZE))
            model.train(EPOCH,BATCH_SIZE,ratings,LR)

            print("Model Loss: {}".format(model.getLoss()))
            print("Running Test")


            # do predictions here
            predictions, losses = model.run_test(100,test_ratings)

            # now we recommend something
            # print(type(test_ratings))
            # print(test_ratings)
            print("Predictions:")           
            # print(type(predictions))
            print(predictions.shape)

            # As a first draft, do just top 5.
            
            # print(max(df_test.user_id.unique()))
            # print(min(df_test.user_id.unique()))
            # userIndex = list(range(min(df_test.user_id.unique()),max(df_test.user_id.unique())))
            # print(userIndex)
            # # print(predictions.argsort(axis = 0)[:5])

            # print("Taking top 5")
            # savedPredictions = predictions.argsort(axis = 0)[:5]
            # print(savedPredictions.shape)
            # topMovies = pd.DataFrame(data = savedPredictions, index=userIndex.shape[0])
            # print("Saving:")
            # np.savetxt("results.csv", topMovies, delimiter="\t")
            


# RunTask1() # uncomment to run task 1
# RunTask2() # Uncomment to run task 2
RunTask3() # uncomment to run task 3