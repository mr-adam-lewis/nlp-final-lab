
# coding: utf-8

# In[3]:

import json, re, gensim, nltk
from sklearn.linear_model import Perceptron
from sklearn.feature_extraction import DictVectorizer
from itertools import izip
import numpy as np


# In[4]:

# Tokenizes the given sentence
def tokenize(sentence):
    return re.findall("\w+", sentence)


# In[5]:

# Converts the given label to an integer representation
def label_to_i(label):
    if label == 'contradiction':
        return -1
    elif label == 'entailment':
        return 1
    else:
        return 0


# In[6]:

# Gathers the data from the files in a structured format
def gather_data(file_path):
    data = [] # all the data entries in the file
    with open(file_path) as f: # open file
        for line in f: # read line by line
            raw = json.loads(line)
            data.append((tokenize(raw['sentence1']), tokenize(raw['sentence2']), label_to_i(raw['gold_label']))) # convert to dict and add to data list
    return data


# In[7]:

# gather the training and testing data in json form from the file
training_data = gather_data('snli_1.0/snli_1.0_train.jsonl')
testing_data = gather_data('snli_1.0/snli_1.0_test.jsonl')
word2vec_model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300-small.bin', binary=True)


# In[8]:

def pairwise(iterable):
    a = iter(iterable)
    return izip(a, a)


# In[9]:

# Takes a sentence as a list of tokens and converts them to a 300 length feature vector based on a word2vec model
def sentence_to_vec(sentence, w2v_model):
    # start with zeros
    sentence_vec = np.zeros(300)
    for token in sentence:
        if token in w2v_model:
            # add each word vector to the sentence vector
            sentence_vec = np.add(sentence_vec, w2v_model[token])
    return sentence_vec


# In[10]:

# Creates a list of cross-unigrams
def create_cross_unigrams(data):
    cross_unigrams = []
    for d in data:
        dic = {}
        for p in d[0]:
            for h in d[1]:
                dic[p+'_'+h] = 1.
        cross_unigrams.append(dic)
    return cross_unigrams


# In[11]:

# Creates a dict of cross-bigrams
def create_cross_bigrams(data):
    cross_bigrams = []
    for d in data:
        dic = {}
        for p1, p2 in pairwise(d[0]):
            for h1, h2 in pairwise(d[1]):
                dic[p1+'_'+p2+'_'+h1+'_'+h2] = 1.
    return cross_bigrams


# In[20]:

def build_feature(data, word2vec_model):
    dic = {}
    # Cross unigrams
    for p in data[0]:
        for h in data[1]:
            dic[p+'_'+h] = 1.

    # Cross bigrams
#     for p1, p2 in pairwise(data[0]):
#         for h1, h2 in pairwise(data[1]):
#             dic[p1+'_'+p2+'_'+h1+'_'+h2] = 1.
            
    # Word2vec
#     premise_vec = sentence_to_vec(data[0], word2vec_model)
#     hypothesis_vec = sentence_to_vec(data[1], word2vec_model)
#     difference_vec = np.subtract(premise_vec, hypothesis_vec)
    
#     for i in range(300):
#         dic['premise-hypothesis-difference-' + str(i)] = difference_vec[i]
    
    return dic

def build_features(data, word2vec_model, v, train_or_test = 'test'):
    features = []
    for d in data:
        features.append(build_feature(d, word2vec_model))
    
    # vectorize the feature dictionaries
    if train_or_test == 'train':
        return (v.fit_transform(features), [data[x][2] for x in range(len(data))])
    else:
        return (v.transform(features), [data[x][2] for x in range(len(data))])


# In[13]:

def test_model(training_data, testing_data, word2vec_model):
    v = DictVectorizer()
    train_features, train_labels = build_features(training_data, word2vec_model, v, 'train')
    test_features, test_labels = build_features(testing_data, word2vec_model, v)
    
    # create the perceptron model
    model = Perceptron(n_iter = 5)
    # fit the model to the training data
    model.fit(train_features, train_labels)
    # get the accuracy on the testing data
    accuracy = model.score(test_features, test_labels)

    return accuracy


# In[14]:

# accuracy = test_model(training_data[:2000], testing_data, word2vec_model)
# print accuracy


# In[21]:

import matplotlib.pyplot as plt

accuracies = []
for i in range(10):
    accuracies.append(test_model(training_data[:300 * (i+1)], testing_data, word2vec_model))

plt.plot([(x+1) * 300 for x in range(10)], accuracies)
plt.ylabel('Accuracy of classifier')
plt.xlabel('Length of training data')
plt.show()


# In[ ]:



