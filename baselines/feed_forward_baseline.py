#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import f1_score

X_test,X_train,X_valid,_,_,_ = pickle.load(file('data.pickle','rb')) 
Y_test,Y_train,Y_valid = pickle.load(file('one_hot_test_labels.pickle','rb'))

print 'Ingested data.'

hidden_size = 10
label_size = 2
dropout_rate = 0.9
input_size = 1364
batch_size = 100
l2 = 0.001
lr = 0.001

start = 0

print 'data loaded.'

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def add_model(X):
	with tf.variable_scope("layer2"):
		U = tf.get_variable("weight2",  (hidden_size, label_size), 
      	initializer=tf.contrib.layers.xavier_initializer())
      	b2 = tf.get_variable("bias2",  (label_size), 
        initializer=tf.zeros)

	with tf.variable_scope("layer1"):
		W = tf.get_variable("weight1",  (input_size, hidden_size), 
        initializer=tf.contrib.layers.xavier_initializer())
      	b1 = tf.get_variable("bias1",  (hidden_size,), 
        initializer=tf.zeros)

	h = tf.tanh(tf.matmul(X,W)+b1)
	tf.add_to_collection("losses", l2*(tf.nn.l2_loss(W)+tf.nn.l2_loss(U)))
	h_drop = tf.nn.dropout(h, dropout_rate)
    # print "h_drop"
	output = tf.matmul(h_drop,U)+b2
	return output 



X = tf.placeholder(tf.float32, shape=(None, input_size))
Y = tf.placeholder(tf.float32, shape=(None, label_size))

h = add_model(X)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(h, Y)) + tf.get_collection("losses")[0]

opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

predict_op = tf.nn.softmax(h)


with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(1):
        for start in range(0, len(X_train), batch_size):
        	print start 
        	feed = {X: X_train[start:start+batch_size,:], Y: Y_train[:,start:start+batch_size].T}
         	[loss_train, prediction,_]=sess.run([loss, predict_op,train_op], feed_dict=feed)
        	print "Training"
        	# print str(loss)
        	ytrain = np.argmax(Y_train[:,start:start+batch_size].T, axis=1)
        	yhattrain = np.argmax(prediction, axis=1)
        	# print prediction
        	# print ytrain
        	# print yhattrain
        	pred_loss_train = f1_score(ytrain,yhattrain)
        	print(i, loss_train, pred_loss_train)
            
        print "Testing"
        feed = {X: X_test, Y: Y_test.T}
       	[prediction]=sess.run([predict_op], feed_dict=feed)

       	ytest = np.argmax(Y_test.T, axis=1)
        yhattest = np.argmax(prediction, axis=1)
        loss_test = np.sum(ytest == yhattest)/float(len(ytest))
        # print prediction

        pred_loss_test = f1_score(ytest,yhattest)

       	print(i, loss_test, pred_loss_test)
