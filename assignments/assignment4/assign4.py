"""
sample code for assign4.py

load_sst can be used to read the files from sst, which can be downloaded from this link:

  https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
  
load_embeddings can be used to read files in the text format. Here's a link to

  word2vec - https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
  GloVe (300D 6B) - http://nlp.stanford.edu/data/glove.840B.300d.zip

The word2vec file is saved in a binary format and will need to be converted to text format. This can be done by installing gensim:

  pip install --upgrade gensim
  
Then running this snippet:

  from gensim.models.keyedvectors import KeyedVectors

  model = KeyedVectors.load_word2vec_format('path/to/GoogleNews-vectors-negative300.bin', binary=True)
  model.save_word2vec_format('path/to/GoogleNews-vectors-negative300.txt', binary=False)

To train:

  python assign4.py

To write test predictions:

  python assign4.py --eval_only_mode

"""


import argparse
import os
import sys
import json
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import nltk
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

import pandas as pd


PAD_TOKEN = '_PAD_'
UNK_TOKEN = '_UNK_'


mydir = os.path.dirname(os.path.abspath(__file__))


# Methods for loading SST data

def sentiment2label(x):
  if x >= 0 and x <= 0.2:
    return 0
  elif x > 0.2 and x <= 0.4:
    return 1
  elif x > 0.4 and x <= 0.6:
    return 2
  elif x > 0.6 and x <= 0.8:
    return 3
  elif x > 0.8 and x <= 1:
    return 4
  else:
    raise ValueError('Improper sentiment value {}'.format(x))


def lemmatizeWord(word,lemmatizer,position='n'):
    #this function lemmatizes the word given
    
    newWord = lemmatizer.lemmatize(word,pos=position)
    
    if position == 'n':
        nextPosition = 'v'
    elif position == 'v':
        nextPosition = 'a'
    elif position == 'a':
        nextPosition = 'r'
    else:
        return newWord
    
    if newWord == word:
        #nothing changed, try something else
        newWord = lemmatizeWord(word,lemmatizer,nextPosition)
        
    
    return newWord

def removeStopWords(wordVector):
    noStop = list(filter(lambda x: x not in nltk.corpus.stopwords.words('english'), wordVector))
#     print(noStop)
    return noStop

# This function will do all of the stemming, lemmatization,
# removal of stop words and other things required for the generation of vocab
def tokenizePhrase(phrase):

    # need to make sure that the output tokens 
    # is as large as the embeddings so that we don't have kernel errors

    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    
#     print(phrase)
    stop_words = set(stopwords.words('english'))
    phraseWords = phrase.split(' ')

    # returnTokens = [] 
    # for word in phraseWords:
    #     newWord = lemmatizeWord(word,lemmatizer)
    #     returnTokens.append(newWord)
        
    # # print(returnTokens)
    # returnTokens = removeStopWords(returnTokens)
        
    return phraseWords

def read_dictionary_txt_with_phrase_ids(dictionary_path, phrase_ids_path, labels_path=None):
  print('Reading data dictionary_path={} phrase_ids_path={} labels_path={}'.format(
    dictionary_path, phrase_ids_path, labels_path))

  with open(phrase_ids_path) as f:
    phrase_ids = set(line.strip() for line in f)

  with open(dictionary_path) as f:
    examples_dict = dict()
    for line in f:
      parts = line.strip().split('|')
      phrase = parts[0]
      phrase_id = parts[1]

      if phrase_id not in phrase_ids:
        continue

      example = dict()
      example['phrase'] = phrase.replace('(', '-LRB').replace(')', '-RRB-')
      example['tokens'] = tokenizePhrase(example['phrase'])
      example['example_id'] = phrase_id
      example['label'] = None
      examples_dict[example['example_id']] = example

  if labels_path is not None:
    with open(labels_path) as f:
      for i, line in enumerate(f):
        if i == 0:
          continue
        parts = line.strip().split('|')
        phrase_id = parts[0]
        sentiment = float(parts[1])
        label = sentiment2label(sentiment)

        if phrase_id in examples_dict:
          examples_dict[phrase_id]['label'] = label

  examples = [ex for _, ex in examples_dict.items()]

  print('Found {} examples.'.format(len(examples)))

  return examples


def build_vocab(datasets):
  vocab = dict()
  vocab[PAD_TOKEN] = len(vocab)
  vocab[UNK_TOKEN] = len(vocab)
  for data in datasets:
    for example in data:
      for word in example['tokens']:
        if word not in vocab:
          vocab[word] = len(vocab)

  print('Vocab size: {}'.format(len(vocab)))

  return vocab


class TokenConverter(object):
  def __init__(self, vocab):
    self.vocab = vocab
    self.unknown = 0

  def convert(self, token):
    if token in self.vocab:
      id = self.vocab.get(token)
    else:
      id = self.vocab.get(UNK_TOKEN)
      self.unknown += 1
    return id


def convert2ids(data, vocab):
  converter = TokenConverter(vocab)
  for example in data:
    example['tokens'] = list(map(converter.convert, example['tokens']))
  print('Found {} unknown tokens.'.format(converter.unknown))
  return data

def concatenateVocabsEmbeddings(v1, e1, v2, e2):


  #compare vocabs,
  # for any word common to both,
  # 
  # concatenate both embeddings for that word ([e1['word'] e2['word']])
  # return combined vocab and embedings
  # print(v1)
  # print(v2)
  commonVocab = set(v1.keys()) & set(v2.keys())
  newEmbedings = np.ndarray(shape=(len(commonVocab),len(e1[0]) + len(e2[0])))
  
  # print(commonVocab)
  i = 0
  for word in commonVocab:
    id1 = v1[word]
    id2 = v2[word]
    # print(id1)
    # print(id2)
    newEmbedings[i] = np.concatenate((e1[id1], e2[id2]))
    i = i + 1

  # now make the commonVocab back into a dictionary with keys = word and value = index (starting at 0)
  generatedVocab = dict((word,index) for word in commonVocab for index in range(len(commonVocab)))

  return generatedVocab, newEmbedings


def load_data_and_embeddings(data_path, phrase_ids_path, embeddings_path):
  labels_path = os.path.join(data_path, 'sentiment_labels.txt')
  dictionary_path = os.path.join(data_path, 'dictionary.txt')
  train_data = read_dictionary_txt_with_phrase_ids(dictionary_path, os.path.join(phrase_ids_path, 'phrase_ids.train.txt'), labels_path)
  validation_data = read_dictionary_txt_with_phrase_ids(dictionary_path, os.path.join(phrase_ids_path, 'phrase_ids.dev.txt'), labels_path)
  test_data = read_dictionary_txt_with_phrase_ids(dictionary_path, os.path.join(phrase_ids_path, 'phrase_ids.test.txt'))
  vocab = build_vocab([train_data, validation_data, test_data])

  if options.useEmbeddingNumber == 1:
    # use embedding for glove
    vocab, embeddings = load_embeddings(options.embeddings, vocab, cache=True)
  elif options.useEmbeddingNumber == 2:
    # use embedding for word2vec (google)
    vocab, embeddings = load_embeddings(options.embeddings2, vocab, cache=True)
  elif options.useEmbeddingNumber == 3:
    # if we are doing double embedding then run it on both and concatenate the results
    vocab1, embeddings1 = load_embeddings(options.embeddings, vocab, cache=True)
    vocab2, embeddings2 = load_embeddings(options.embeddings2, vocab, cache=True)

    #Merge the vocabs and embeddings
    #append ----300 Glove ---- ---- 300 word2vec ----
    vocab, embeddings = concatenateVocabsEmbeddings(vocab1, embeddings1, vocab2, embeddings2)

  else:
    print("Embedding mode not support, please specify one of the options (1,2,3), reverting to standard embeddings:")
    vocab, embeddings = load_embeddings(options.embeddings, vocab, cache=True)


  train_data = convert2ids(train_data, vocab)
  validation_data = convert2ids(validation_data, vocab)
  test_data = convert2ids(test_data, vocab)
  return train_data, validation_data, test_data, vocab, embeddings


def load_embeddings(path, vocab, cache=False, cache_path=None):
  rows = []
  new_vocab = [UNK_TOKEN]

  if cache_path is None:
    cache_path = path + '.cache'

  # Use cache file if it exists.
  if os.path.exists(cache_path):
    path = cache_path

  print("Reading embeddings from {}".format(path))

  # first pass over the embeddings to vocab and relevant rows
  with open(path) as f:
    for line in f:
      word, row = line.split(' ', 1)
      if word == UNK_TOKEN:
        raise ValueError('The unk token should not exist w.in embeddings.')
      if word in vocab:
        rows.append(line)
        new_vocab.append(word)

  # optionally save relevant rows to cache file.
  if cache and not os.path.exists(cache_path):
    with open(cache_path, 'w') as f:
      for line in rows:
        f.write(line)
      print("Cached embeddings to {}".format(cache_path))

  # turn vocab list into a dictionary
  new_vocab = {w: i for i, w in enumerate(new_vocab)}

  print('New vocab size: {}'.format(len(new_vocab)))

  assert len(rows) == len(new_vocab) - 1

  # create embeddings matrix
  embeddings = np.zeros((len(new_vocab), 300), dtype=np.float32)
  for i, line in enumerate(rows):
    embeddings[i+1] = list(map(float, line.strip().split(' ')[1:]))

  return new_vocab, embeddings


# Batch Iterator

def prepare_data(data,maxKernelSize):
  # pad data
  maxlen = max(map(len, data))
  if (maxlen < maxKernelSize):
    maxlen = maxKernelSize

  data = [ex + [0] * (maxlen-len(ex)) for ex in data]

  # wrap in tensor
  if torch.cuda.is_available():
    return torch.cuda.LongTensor(data)
  else:
    return torch.LongTensor(data)


def prepare_labels(labels):
  try:
    if torch.cuda.is_available():
      return torch.cuda.LongTensor(labels)
    else:
      return torch.LongTensor(labels)
  except:
    return labels


def batch_iterator(dataset, batch_size, forever=True, maxEpochs=200):
  dataset_size = len(dataset)
  order = None
  nbatches = dataset_size // batch_size

  def init_order():
    return random.sample(range(dataset_size), dataset_size)

  def get_batch(start, end):
    batch = [dataset[ii] for ii in order[start:end]]
    data = prepare_data([ex['tokens'] for ex in batch],6)
    labels = prepare_labels([ex['label'] for ex in batch])
    example_ids = [ex['example_id'] for ex in batch]
    return data, labels, example_ids

  order = init_order()
  numEpochs = 0

  # print("# Batches = " + str(nbatches))
  # print("Batch Size = " + str(batch_size))

  while True:
  # while numEpochs < maxEpochs:
    # print("Epoch = " + str(numEpochs))
    for i in range(nbatches):

      start = i*batch_size
      end = (i+1)*batch_size
      yield get_batch(start, end)

    # numEpochs += 1


    if nbatches*batch_size < dataset_size:
      yield get_batch(nbatches*batch_size, dataset_size)

    if not forever:
      break
    
    order = init_order()

# Models

class BagOfWordsModel(nn.Module):
  def __init__(self, embeddings):
    super(BagOfWordsModel, self).__init__()
    self.embed = nn.Embedding(embeddings.shape[0], embeddings.shape[1], sparse=True)
    self.embed.weight.data.copy_(torch.from_numpy(embeddings))
    self.classify = nn.Linear(embeddings.shape[1], 5)

  def forward(self, x):
    return self.classify(self.embed(x).sum(1))


# Train a CNN Sentiment Classifier using pre-trained
# and frozen skip-gram word2vec as your word embedding
# on the Sentiment data from above
class CNNClassifier(nn.Module):



    def __init__(self, vocab, embeddings, output_size, kernel_dim=100, kernel_sizes=(3, 4, 5), dropout=0.5):
        super(CNNClassifier,self).__init__()

        # kernel_sizes should now be working
        print("Embedding Size: " + str(embeddings.shape[0]))
        print("Embedding Size: " + str(embeddings.shape[1]))

        embeddingsPosition = 1


        self.embedding = nn.Embedding(len(vocab), embeddings.shape[embeddingsPosition])
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embeddings.shape[embeddingsPosition])) for K in kernel_sizes])

        # kernal_size = (K,D) 
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, output_size)
    
    
    def init_weights(self, pretrained_word_vectors, is_static=False): # finetuned with is_static to true
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if is_static:
            self.embedding.weight.requires_grad = False
            


    def forward(self, inputs, is_training=False):
        inputs = self.embedding(inputs).unsqueeze(1) # (B,1,T,D)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs] #[(N,Co,W), ...]*len(Ks)
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs] #[(N,Co), ...]*len(Ks)

        concated = torch.cat(inputs, 1)

        if is_training:
            concated = self.dropout(concated) # (N,len(Ks)*Co)
        out = self.fc(concated) 
        return F.log_softmax(out,1)

  


# Utility Methods

def checkpoint_model(step, val_err, model, opt, save_path):
  save_dict = dict(
    step=step,
    val_err=val_err,
    model_state_dict=model.state_dict(),
    opt_state_dict=opt.state_dict())
  torch.save(save_dict, save_path)


def load_model(model, opt, load_path):
  load_dict = torch.load(load_path)
  step = load_dict['step']
  val_err = load_dict['val_err']
  model.load_state_dict(load_dict['model_state_dict'])
  opt.load_state_dict(load_dict['opt_state_dict'])
  return step, val_err


# Main

def run_validation(model, dataset, options):
  err = 0
  count = 0

  numPerLabels = []
  numCorrectPredLabels = []

  # making the run_validation print out the stats for each label
  for data, labels, _ in batch_iterator(dataset, options.batch_size, forever=False):
    outp = model(Variable(data))
    loss = nn.NLLLoss()(F.log_softmax(outp), Variable(labels))
    acc = (outp.data.max(1)[1] == labels).sum() / data.shape[0]
    err += (1-acc) * data.shape[0]
    count += data.shape[0]

    #get the stats per label
    # print(labels.shape)
    # print(outp.data.max(1)[1].shape)


  err = err / count
  print('Ev-Err={}'.format(err))

  return err


def run_test(model, dataset, options):
  print('Writing predictions to {}'.format(os.path.abspath(options.predictions)))

  preds_dict = dict()

  for data, _, example_ids in batch_iterator(dataset, options.batch_size, forever=False):
    outp = model(Variable(data))
    preds = outp.data.max(1)[1]

    for id, pred in zip(example_ids, preds):
      preds_dict[id] = pred

  with open(options.predictions, 'w') as f:
    for id, pred in preds_dict.items():
      f.write('{}|{}\n'.format(id, pred))


def run(options):
  train_data, validation_data, test_data, vocab, embeddings = \
    load_data_and_embeddings(options.data, options.ids, options.embeddings)
  model = CNNClassifier(vocab, embeddings, 5)

  if (options.fineTuned):
    model.init_weights(embeddings,is_static=True) # finetuned we want the weights to be fixed except for the last ones
  else:
    model.init_weights(embeddings,is_static=False) # not-finetuned we want everything to not be fixed


  USE_CUDA = torch.cuda.is_available()
  if USE_CUDA:
    torch.cuda.set_device(0) # just use the first gpu available.

  FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
  LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
  ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor
  
  if USE_CUDA:
      model = model.cuda()

  opt = optim.SGD(model.parameters(), lr=3e-4)
  
  step = 0
  best_val_err = 1

  if options.eval_only_mode:
    step, best_val_err = load_model(model, opt, options.model)
    print('Model loaded from {}\nstep={} best_val_err={}'.format(options.model, step, best_val_err))
    run_test(model, test_data, options)
    # sys.exit()
    # maybe not exit, instead just return out of the run function
    return

  print("Entering batch iterator loop")
  for data, labels, _ in batch_iterator(train_data, options.batch_size, forever=True):
    outp = model(Variable(data))
    loss = nn.NLLLoss()(F.log_softmax(outp), Variable(labels))
    acc = (outp.data.max(1)[1] == labels).sum() / data.shape[0]

    opt.zero_grad()
    loss.backward()
    opt.step()
    
    if step % options.log_every == 0:
      print('Step={} Tr-Loss={} Tr-Acc={}'.format(step, loss.data[0], acc))
      
    if step % options.eval_every == 0:
      val_err = run_validation(model, validation_data, options)
      
      # early stopping
      if val_err < best_val_err:
        best_val_err = val_err
        print('Checkpointing model step={} best_val_err={}.'.format(step, best_val_err))
        checkpoint_model(step, val_err, model, opt, options.model)

    if step == options.maxNumSteps:
      print('Completed Training with step={} best_val_err={}.'.format(step, best_val_err))
      # checkpoint_model(step, val_err, model, opt, options.model)
      break
    
    step += 1


if __name__ == '__main__':

  # Set a seed for numpy, pytorch
  np.random.seed(0)
  torch.manual_seed(0)

  if torch.cuda.is_available():
    print("Cuda is Available, will try and use it")
  else:
    print("Cuda not available, not able to use")


# Only glove
# create the CNN and train
  parser = argparse.ArgumentParser()
  parser.add_argument('--ids', default=mydir, type=str)
  parser.add_argument('--data', default=os.path.expanduser('data/stanfordSentimentTreebank'), type=str)
  parser.add_argument('--embeddings', default=os.path.expanduser('data/glove/glove.840B.300d.txt'), type=str)
  parser.add_argument('--model', default=os.path.join(mydir, 'model-cnnTweaked-glove.ckpt'), type=str)
  parser.add_argument('--predictions', default=os.path.join(mydir, 'predictions-cnnTweaked-glove.txt'), type=str)
  parser.add_argument('--log_every', default=500, type=int)
  parser.add_argument('--eval_every', default=4000, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--eval_only_mode', action='store_true')

  parser.add_argument('--embeddings2', default=os.path.expanduser('data/GoogleNews-vectors-negative300.txt'), type = str)
  parser.add_argument('--useEmbeddingNumber', default=1, type = int) # this takes value 1, 2 or 3 (1 = glove, 2 = word2vec, 3 = both)
  parser.add_argument('--usePreProcess', default=False, type=bool) # this determines if we use preprocessing / finetuning on the model

  parser.add_argument('--maxNumSteps', default = 8000, type = int)

  parser.add_argument('--fineTuned', default = False, type = bool) # make this true for finetuned, false otherwise
  # parser.add_argument('--kernel_sizes', default=(2,3,4), type = [int])



  options = parser.parse_args()

  print(json.dumps(options.__dict__, sort_keys=True, indent=4))

  run(options)

  ## now generate predictions for this classifier.

  parser = argparse.ArgumentParser()
  parser.add_argument('--ids', default=mydir, type=str)
  parser.add_argument('--data', default=os.path.expanduser('data/stanfordSentimentTreebank'), type=str)
  parser.add_argument('--embeddings', default=os.path.expanduser('data/glove/glove.840B.300d.txt'), type=str)
  parser.add_argument('--model', default=os.path.join(mydir, 'model-cnnTweaked-glove.ckpt'), type=str)
  parser.add_argument('--predictions', default=os.path.join(mydir, 'predictions-cnnTweaked-glove.txt'), type=str)
  parser.add_argument('--log_every', default=500, type=int)
  parser.add_argument('--eval_every', default=4000, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--eval_only_mode', action='store_false')

  parser.add_argument('--embeddings2', default=os.path.expanduser('data/GoogleNews-vectors-negative300.txt'), type = str)
  parser.add_argument('--useEmbeddingNumber', default=1, type = int) # this takes value 1, 2 or 3 (1 = glove, 2 = word2vec, 3 = both)
  parser.add_argument('--usePreProcess', default=False, type=bool) # this determines if we use preprocessing / finetuning on the model

  parser.add_argument('--maxNumSteps', default = 8000, type = int)
  parser.add_argument('--fineTuned', default = False, type = bool) # make this true for finetuned, false otherwise


  options = parser.parse_args()
  run(options)
# only word2vec

  # Set a seed for numpy, pytorch
  np.random.seed(0)
  torch.manual_seed(0)


  parser = argparse.ArgumentParser()
  parser.add_argument('--ids', default=mydir, type=str)
  parser.add_argument('--data', default=os.path.expanduser('data/stanfordSentimentTreebank'), type=str)
  parser.add_argument('--embeddings', default=os.path.expanduser('data/glove/glove.840B.300d.txt'), type=str)
  parser.add_argument('--model', default=os.path.join(mydir, 'model-cnnTweaked-word2vec.ckpt'), type=str)
  parser.add_argument('--predictions', default=os.path.join(mydir, 'predictions-cnnTweaked-word2vec.txt'), type=str)
  parser.add_argument('--log_every', default=500, type=int)
  parser.add_argument('--eval_every', default=4000, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--eval_only_mode', action='store_true')

  parser.add_argument('--embeddings2', default=os.path.expanduser('data/GoogleNews-vectors-negative300.txt'), type = str)
  parser.add_argument('--useEmbeddingNumber', default=2, type = int) # this takes value 1, 2 or 3 (1 = glove, 2 = word2vec, 3 = both)
  parser.add_argument('--usePreProcess', default=False, type=bool) # this determines if we use preprocessing / finetuning on the model


  parser.add_argument('--maxNumSteps', default = 8000, type = int)

  parser.add_argument('--fineTuned', default = False, type = bool) # make this true for finetuned, false otherwise


  options = parser.parse_args()

  print(json.dumps(options.__dict__, sort_keys=True, indent=4))

  run(options)

  ## now generate predictions for this classifier.

  parser = argparse.ArgumentParser()
  parser.add_argument('--ids', default=mydir, type=str)
  parser.add_argument('--data', default=os.path.expanduser('data/stanfordSentimentTreebank'), type=str)
  parser.add_argument('--embeddings', default=os.path.expanduser('data/glove/glove.840B.300d.txt'), type=str)
  parser.add_argument('--model', default=os.path.join(mydir, 'model-cnnTweaked-word2vec.ckpt'), type=str)
  parser.add_argument('--predictions', default=os.path.join(mydir, 'predictions-cnnTweaked-word2vec.txt'), type=str)
  parser.add_argument('--log_every', default=500, type=int)
  parser.add_argument('--eval_every', default=4000, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--eval_only_mode', action='store_false')

  parser.add_argument('--embeddings2', default=os.path.expanduser('data/GoogleNews-vectors-negative300.txt'), type = str)
  parser.add_argument('--useEmbeddingNumber', default=2, type = int) # this takes value 1, 2 or 3 (1 = glove, 2 = word2vec, 3 = both)
  parser.add_argument('--usePreProcess', default=False, type=bool) # this determines if we use preprocessing / finetuning on the model


  parser.add_argument('--maxNumSteps', default = 8000, type = int)
  parser.add_argument('--fineTuned', default = False, type = bool) # make this true for finetuned, false otherwise



  options = parser.parse_args()
  run(options)

# both word2vec and glove

  # Set a seed for numpy, pytorch
  np.random.seed(0)
  torch.manual_seed(0)

  parser = argparse.ArgumentParser()
  parser.add_argument('--ids', default=mydir, type=str)
  parser.add_argument('--data', default=os.path.expanduser('data/stanfordSentimentTreebank'), type=str)
  parser.add_argument('--embeddings', default=os.path.expanduser('data/glove/glove.840B.300d.txt'), type=str)
  parser.add_argument('--model', default=os.path.join(mydir, 'model-cnnTweaked-both.ckpt'), type=str)
  parser.add_argument('--predictions', default=os.path.join(mydir, 'predictions-cnnTweaked-both.txt'), type=str)
  parser.add_argument('--log_every', default=500, type=int)
  parser.add_argument('--eval_every', default=4000, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--eval_only_mode', action='store_true')

  parser.add_argument('--embeddings2', default=os.path.expanduser('data/GoogleNews-vectors-negative300.txt'), type = str)
  parser.add_argument('--useEmbeddingNumber', default=3, type = int) # this takes value 1, 2 or 3 (1 = glove, 2 = word2vec, 3 = both)
  parser.add_argument('--usePreProcess', default=False, type=bool) # this determines if we use preprocessing / finetuning on the model


  parser.add_argument('--maxNumSteps', default = 8000, type = int)
  parser.add_argument('--fineTuned', default = False, type = bool) # make this true for finetuned, false otherwise


  options = parser.parse_args()

  print(json.dumps(options.__dict__, sort_keys=True, indent=4))

  run(options)

  parser = argparse.ArgumentParser()
  parser.add_argument('--ids', default=mydir, type=str)
  parser.add_argument('--data', default=os.path.expanduser('data/stanfordSentimentTreebank'), type=str)
  parser.add_argument('--embeddings', default=os.path.expanduser('data/glove/glove.840B.300d.txt'), type=str)
  parser.add_argument('--model', default=os.path.join(mydir, 'model-cnnTweaked-both.ckpt'), type=str)
  parser.add_argument('--predictions', default=os.path.join(mydir, 'predictions-cnnTweaked-both.txt'), type=str)
  parser.add_argument('--log_every', default=500, type=int)
  parser.add_argument('--eval_every', default=4000, type=int)
  parser.add_argument('--batch_size', default=64, type=int)
  parser.add_argument('--eval_only_mode', action='store_false')

  parser.add_argument('--embeddings2', default=os.path.expanduser('data/GoogleNews-vectors-negative300.txt'), type = str)
  parser.add_argument('--useEmbeddingNumber', default=3, type = int) # this takes value 1, 2 or 3 (1 = glove, 2 = word2vec, 3 = both)
  parser.add_argument('--usePreProcess', default=False, type=bool) # this determines if we use preprocessing / finetuning on the model


  parser.add_argument('--maxNumSteps', default = 8000, type = int)
  parser.add_argument('--fineTuned', default = False, type = bool) # make this true for finetuned, false otherwise

  ## now generate predictions for this classifier.

  options = parser.parse_args()

  run(options)