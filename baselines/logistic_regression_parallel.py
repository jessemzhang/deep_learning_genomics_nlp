import numpy as np
import pickle
import time
import os

X_test,X_train,Y_test,Y_train = pickle.load(file('data_100.pickle','rb'))

from sklearn import linear_model
from sklearn.metrics import f1_score

def logistic_reg(inputs):
    X_test,X_train,Y_test,Y_train,i = inputs
    print i
    if str(i) not in os.listdir('logistic_regression_100_results/'):
        f = open('logistic_regression_100_results/'+str(i),'w')
        if np.sum(Y_test) > 0 and np.sum(Y_train) > 0:
            logreg = linear_model.LogisticRegression(C=1e6)
            logreg.fit(X_train, Y_train)
            Yhat_train = logreg.predict(X_train)
            Yhat_test = logreg.predict(X_test)
            f1_train = f1_score(Y_train,Yhat_train)
            f1_test = f1_score(Y_test,Yhat_test)
            f.write('%d\t%d\t%.3f\t%.3f\n'%(np.sum(Y_train),np.sum(Y_test),f1_train,f1_test))
            f.close()
            return f1_train,f1_test
        else:
            f.write('%d\t%d\n'%(np.sum(Y_train),np.sum(Y_test)))
            f.close()

import multiprocessing as mp

num_tasks = np.shape(Y_train)[1]
all_inputs = [(X_test,X_train,Y_test[:,i],Y_train[:,i],i) for i in range(num_tasks)]
pool=mp.Pool(processes=64)
pool.map(logistic_reg,all_inputs)
