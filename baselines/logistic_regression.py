
# coding: utf-8

# In[3]:

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import f1_score
import pickle
import time

# Test different feature vectors for each sequence
d = {'A':0,'G':1,'C':2,'T':3}

# Construct k-mer dictionary
kmer_to_ind = {}
ind_to_kmer = {}
k_ind = 0
for k in range(1,6):
    for kmer in [''.join(i) for i in itertools.product('ACGT', repeat = k)]:
        kmer_to_ind[kmer] = k_ind
        ind_to_kmer[k_ind] = kmer
        k_ind += 1

# Feature mapping 1: char to int
def seq_to_int(s):
    return map(lambda x: d[x], s)


# Feature mapping 2: k-mer counting for k = 1, 2, 3, 4, 5
def kmer_count(s):
    v = np.zeros(len(kmer_to_ind))
    for k in range(1,6):
        for kmer in [s[i:i+k] for i in range(len(s)-k+1)]:
            v[kmer_to_ind[kmer]] += 1
    return v


# In[7]:



def load_features(feature_extractor,filename):
    # Read in sequence
    f = open(filename)
    X = []
    start_time = time.time()
    for i,line in enumerate(f):
        if i % 1000 == 0:
            print time.time()-start_time, ' s'
        X.append(feature_extractor(line.split()[0]))
    return np.array(X)
  
fileprefix = '../jz-rnn-tensorflow/data/deepsea'
feature_extractor = kmer_count
# fileprefix = '../jz-rnn-tensorflow/data/positive/positive'
# X_test = load_features(feature_extractor,fileprefix + '.data.test.txt')
# X_train = load_features(feature_extractor,fileprefix + '.data.train.txt')
# # X_train = load_features(feature_extractor,'../deepsea_train/deepsea_feature915.data.train.txt')
# X_valid = load_features(feature_extractor,fileprefix + '.data.valid.txt')
# Y_test = np.loadtxt(fileprefix + '.labels.test.txt')
# Y_train = np.loadtxt(fileprefix + '.labels.train.txt')
# # Y_train = np.loadtxt('../deepsea_train/deepsea_feature915.labels.train.txt')
# Y_valid = np.loadtxt(fileprefix + '.labels.valid.txt')


# In[8]:

# pickle.dump((X_test,X_train,X_valid,Y_test,Y_train,Y_valid),file('data.pickle','wb'))
X_test,X_train,X_valid,Y_test,Y_train,Y_valid = pickle.load(file('data.pickle','rb')) 

print "data loaded."

# In[ ]:

# logistic regression model


clist = [1,10,100,1000,10000,100000,100000,1000000,10000000]

train_errors = []
test_errors = []

for c in clist:
    print c
    logreg = linear_model.LogisticRegression(C=c)
    logreg.fit(X_train, Y_train)
    Yhat_train = logreg.predict(X_train)
    Yhat_test = logreg.predict(X_test)
    train_errors.append(f1_score(Y_train,Yhat_train))
    test_errors.append(f1_score(Y_test,Yhat_test))
    print train_errors
    print test_errors


# In[4]:

# Look at error rates (F1)
clist = [1,10,100,1000,10000,100000,100000,1000000,10000000]
np.savetxt('logististic_regulariser.txt',clist)
np.savetxt('logististic_train_errors.txt',train_errors)
np.savetxt('logististic_test_errors.txt',test_errors)


# In[ ]:



