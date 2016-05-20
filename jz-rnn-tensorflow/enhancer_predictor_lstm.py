# Predict whether or not an enhancer will bind based on the sequence
# The data used is from the DeepSEA paper and can be downloaded at
#
#     http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz
#
# Written by Jesse Zhang 5/19/2016

from __future__ import division
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf

class Config(object):
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 1000 #(the entire length of the sequence for the dataset)
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
        self.cost = tf.reduce_sum(loss) / batch_size

        # Optimizer
        optimizer = tf.train.AdamOptimizer(config.learning_rate)
        train_op = optimizer.minimize(loss)

    def run_epoch(self, session, data, labels, verbose=True):
        """ 
        Train the model!
        
        Args:
          session: tensorflow session
          data: length N array of sequences
          labels: length N array of labels (integers)
          verbose: self explanatory

        Returns:
          Perplexity for this epoch
        """

        epoch_size = ((len(data) // self.config.batch_size) - 1) // self.config.num_steps
        state_time = time.time()
        costs = 0.0
        iters = 0
        state = self.initial_state.eval()
        for step, (x,y) in enumerate(enhancer_iterator(data, labels, 
                                                       self.config.batch_size,
                                                       self.config.num_steps)):
            cost,state = session.run([ self.cost, self.final_state ],
                                     { self.input_data: x,
                                       self.targets: y,
                                       self.initial_state: state })
            costs += cost
            iters += self.config.num_steps

            if verbose and step % (epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                      (step * 1.0 / epoch_size, np.exp(costs / iters),
                       iters * self.config.batch_size / (time.time() - start_time)))

        return np.exp(costs / iters)
        
    def enhancer_iterator(self, data, batch_size, num_steps):
        """
        Generate batch-size pointers on raw sequence data for minibatch iteration
        
        Args:
          data: length N array of sequences (nucleotides A C T G)
          labels: length N array of labels (integers) for each sequence
                  (the max value of this array = # of labels)
          batch_size: int, the batch size
          num_steps: int, number of unrolls
        
        Yields:
          Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
          The second element of the tuple is the same data 
          
        """
        def seq_to_ints(seq):
            return [self.vocab.word_to_index[c] for c in seq]

        # Map raw data to array of ints. if all sequences are the same length L, 
        # raw_data will be N-by-L
        raw_data = np.array([seq(i) for i in data], dtype=np.int32)
        data_len = len(raw_data)
        batch_len = data_len // batch_size
        # data will have batch_len elements, each of size batch_size
        data = np.zeros([batch_size,batch_len], dtype=np.int32)
        for i in range(batch_size):
            data[i] = raw_data[batch_len*i:batch_len*(i+1)]

        epoch_size = (batch_len-1) // num_steps
        if epoch_size == 0: print("ERROR: decrease batch_size or num_steps")

        for i in range(epoch_size):
            x = data[:,i*num_steps:(i+1)*num_steps]
            y = labels[:,
