import numpy as np
from scipy.sparse import rand as sprand
from scipy.sparse import lil_matrix
import torch
from torch.autograd import Variable
import pandas as pd

def get_movielens_ratings(df):
    n_users = max(df.user_id.unique())
    n_items = max(df.item_id.unique())

    interactions = lil_matrix( (n_users,n_items), dtype=float) #np.zeros((n_users, n_items))
    for row in df.itertuples():
        interactions[row[1] - 1, row[2] - 1] = row[3]
    return interactions

def get_batch(batch_size,ratings):
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

def run_epoch():
    for i,batch in enumerate(get_batch(BATCH_SIZE, ratings)):
        # Set gradients to zero
        reg_loss_func.zero_grad()
        
        # Turn data into variables
        interactions = Variable(torch.FloatTensor(ratings[batch, :].toarray()))
        rows = Variable(torch.LongTensor(batch))
        cols = Variable(torch.LongTensor(np.arange(ratings.shape[1])))
    
        # Predict and calculate loss
        predictions = model(rows, cols)
        loss = loss_func(predictions, interactions)
    
        # Backpropagate
        loss.backward()
    
        # Update the parameters
        reg_loss_func.step()

class MatrixFactorization(torch.nn.Module):
    
    def __init__(self, n_users, n_items, n_factors=5):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, 
                                               n_factors,
                                               sparse=False)
        self.item_factors = torch.nn.Embedding(n_items, 
                                               n_factors,
                                               sparse=False)
        # Also should consider fitting overall bias (self.mu term) and both user and item bias vectors
        # Mu is 1x1, user_bias is 1xn_users. item_bias is 1xn_items
    
    # For convenience when we want to predict a sinble user-item pair. 
    def predict(self, user, item):
        # Need to fit bias factors
        return (pred + self.user_factors(user) * self.item_factors(item)).sum(1)
    
    # Much more efficient batch operator. This should be used for training purposes
    def forward(self, users, items):
        # Need to fit bias factors
        return torch.mm(self.user_factors(users),torch.transpose(self.item_factors(items),0,1))


print("Starting")
names = ['user_id', 'item_id', 'rating', 'timestamp']
df_train = pd.read_csv('ml-10M100K/r3.train', sep='::', names=names,engine='python')
print("Loaded train data")
df_test = pd.read_csv('ml-10M100K/r3.test', sep='::', names=names,engine='python')
print("Loaded Data")

ratings = get_movielens_ratings(df_train)
print(ratings.shape)

test_ratings = get_movielens_ratings(df_test)
print(test_ratings.shape)

print("Prepared the data")


model = MatrixFactorization(ratings.shape[0], ratings.shape[1], n_factors=2)

print("Created the model")
if torch.cuda.is_available():
    model.cuda()
    print("Using CUDA")


print("setting loss function")
loss_func = torch.nn.MSELoss()
reg_loss_func = torch.optim.SGD(model.parameters(), lr=1e-6, weight_decay=1e-3)


print("Running the training")
EPOCH = 2
BATCH_SIZE = 1000 #50
LR = 0.001

for i in range(EPOCH):
    print(i)
    run_epoch()


print("Running Test")

# predict on a single value
predictionsList = [] # this array will be the values of the current run

# for i,batch in enumerate(get_batch(BATCH_SIZE,test_ratings)):
rows = Variable(torch.LongTensor(test_ratings[28665].toarray()[0]))
cols = Variable(torch.LongTensor(test_ratings[:,0].T.toarray()[0]))
predictions = model(rows, cols)
predictionsList.append(predictions.data.cpu().numpy())

print(predictionsList)

