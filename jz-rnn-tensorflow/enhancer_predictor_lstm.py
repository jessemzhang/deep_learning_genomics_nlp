# Predict whether or not an enhancer will bind based on the sequence
# The data used is from the DeepSEA paper and can be downloaded at
#
#     http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz
#
# Written by Jesse Zhang 5/19/2016

import time
import numpy as np
import tensorflow as tf

class Config(object):
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 200
    hidden_size = 128
    max_epoch = 40
    keep_prob = 0.9
    lr_decay = 0.8
    batch_size = 20

class CharLevelDNAVocab(object):
    size = 4
    word_to_index = {'A':0,'G':1,'C':2,'T':3}
    index_to_word = {0:'A',1:'G',2:'C',3:'T'}

class EnhancerRNN(object):
    """ Enhancer prediction model. """
    
    def __init__(self,config,vocab):
        self.config = config
        self.vocab = vocab

        # Placeholder variables
        self.input_data = tf.placeholder(tf.int32, [batch_size,num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size])

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab.size,config.hidden_size])
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)
            inputs = tf.dropout(inputs, config.keep_prob)
        
        # The "Recurrent" part (only look at last output of the sequence)
        cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*config.num_layers)
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for t in range(config.num_steps):
                if t > 0:
                    tf.get_variable_scope().reuse_variables()
                    (output, state) = cell(inputs[:,t,:], state)
        self.final_state = state
                    
        # The prediction (softmax) part
        Ws = tf.get_variable("softmax_w", [config.hidden_size,vocab.size])
        bs = tf.get_variable("softmax_b", [vocab.size])
        logits = tf.matmul(output,Ws) + bs

        # Loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits,self.targets)
        
        # Optimizer
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        train_op = optimizer.minimize(loss)

    def run_epoch(self, session, data, labels, verbose=True):
        """ 
        Train the model!
        
        Args:
          session: tensorflow session
          data: N-by-(sequence-length) matrix
          labels: N-by-(#labels) matrix of one-hot labels
          verbose: self explanatory

        Returns:
          Perplexity for this epoch
        """

        epoch_size = ((len(data) // self.config.batch_size) - 1) // self.config.num_steps
        state_time = time.time()
        costs = 0.0
        iters = 0
        state = self.initial_state.eval()
        for step, (x,y) in enumerate(
        
        
        
        
