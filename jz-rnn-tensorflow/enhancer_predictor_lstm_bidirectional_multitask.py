# Predict whether or not an enhancer will bind based on the sequence
# The data used is from the DeepSEA paper and can be downloaded at
#
#     http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz
#
# Written by Jesse Zhang 5/19/2016
#
# Usage: python enhancer_predictor_lstm.py DATA_PREFIX

from __future__ import division
from __future__ import print_function
import time,sys,os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import *
from sklearn.metrics import f1_score
from jz_rnn_cell import *

class DeepseaConfig(object):
    learning_rate = 1.0e-3
    max_grad_norm = 5
    num_layers = 2
    num_steps = 100 # 1000
    embed_size = 2
    hidden_size = 128
    keep_prob = 0.95
    max_epoch = 200
    epochs_with_same_lr = 2
    lr_decay = 0.5
    batch_size = 20
    num_classes = 2 # per task 
    num_tasks = 919 # for multitask learning
    early_stopping = 100

class PositiveConfig(object):
    learning_rate = 1.0e-3
    max_grad_norm = 5
    num_layers = 2
    num_steps = 10
    embed_size = 2
    hidden_size = 64
    keep_prob = 0.95
    max_epoch = 20
    epochs_with_same_lr = 2
    lr_decay = 0.5
    batch_size = 8 
    num_classes = 2 # per task 
    num_tasks = 3   # for multitask learning
    early_stopping = 40

class CharLevelDNAVocab(object):
    size = 4
    word_to_index = {'A':0,'G':1,'C':2,'T':3}
    index_to_word = {0:'A',1:'G',2:'C',3:'T'}

class EnhancerRNN(object):
    """ Enhancer prediction model. """
    
    def __init__(self,config,vocab):
        self.config = config
        self.vocab = vocab
        self.dropout = config.keep_prob

        # Placeholder variables
        self.input_data = tf.placeholder(tf.int32, [config.batch_size,config.num_steps])
        self.targets = tf.placeholder(tf.int32, [config.batch_size,config.num_tasks])
        self.dropout = tf.placeholder(tf.float32)

        embedding = tf.get_variable("embedding", [vocab.size,config.embed_size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        
        # The "Recurrent" part (only look at last output of the sequence)
        cell = jzGRUCell(config.hidden_size,activation=tf.nn.tanh)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell]*config.num_layers)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                                             input_keep_prob = self.dropout,
                                             output_keep_prob = self.dropout)
        cell_fw = cell
        cell_bw = cell
        self.initial_state = cell.zero_state(self.config.batch_size, tf.float32)
        self.initial_state = tf.concat(1,[self.initial_state,self.initial_state])
        output,state = self.bi_directional_RNN(cell_fw,cell_bw,inputs,self.initial_state)
        self.final_state = state
                    
        # The prediction (softmax) part
        Ws,bs = [],[]
        self.predictions = []
        cost = 0
        for i in range(config.num_tasks): 
            # Same weights for RNN for all tasks; different softmax weights
            Ws.append(tf.get_variable("softmax_w"+str(i), 
                                      [config.hidden_size*2,config.num_classes]))
            bs.append(tf.get_variable("softmax_b"+str(i), [config.num_classes]))
            logits = tf.matmul(output,Ws[i]) + bs[i]
            self.predictions.append(tf.nn.softmax(logits))

            # Loss ("sparse" version of this function requires mutually exclusive classes)
            labels = tf.squeeze(tf.slice(self.targets,[0,i],[config.batch_size,1]))
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels)
            cost += tf.reduce_sum(loss) / config.batch_size
        cost /= config.num_tasks
        self.cost = cost
        
        # Optimizer (learning rate will be assigned using assign_lr)
        self.lr = tf.Variable(0.0, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) 
        self.train_op = optimizer.minimize(cost)

    def one_direction_RNN(self, cell, inputs, state, direction='FW'):
        itersteps = range(self.config.num_steps)
        if direction == 'BW': itersteps.reverse()
        with tf.variable_scope("RNN"+direction):
            for i,t in enumerate(itersteps):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                (output, state) = cell(inputs[:,t,:], state)
        return output,state

    def bi_directional_RNN(self, cell_fw, cell_bw, inputs, initial_state):
        # Use two one_direction_RNN function outputs. Concatenate the output
        initial_state_fw,initial_state_bw = tf.split(1,2,initial_state)
        output_fw,final_state_fw = self.one_direction_RNN(cell_fw,inputs, 
                                                          initial_state_fw, direction='FW')
        output_bw,final_state_bw = self.one_direction_RNN(cell_bw,inputs, 
                                                          initial_state_bw, direction='BW')
        output = tf.concat(1,[output_fw,output_bw])
        final_state = tf.concat(1,[final_state_fw,final_state_bw])
        return output,final_state
        
    def assign_lr(self, session, lr):
        """ Update learning rate if needed (for adaptive learning rates) """
        session.run(tf.assign(self.lr,lr))
        
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
        num_batches = (len(data) // self.config.batch_size)
        start_time = time.time()
        costs = 0.0
        state = self.initial_state.eval()
        for step, (x,y) in enumerate(self.enhancer_iterator(data, labels, 
                                                       self.config.batch_size,
                                                       self.config.num_steps)):
            cost,state,_ = session.run([ self.cost, self.final_state, self.train_op ],
                                       { self.input_data: x,
                                         self.targets: y,
                                         self.dropout: self.config.keep_prob,
                                         self.initial_state: state })
            costs += cost

            if verbose and num_batches // 10 == 0: # In case num_batches < 10
                print("batch %d of %d; perplexity: %.3f; time elapsed: %.3f s" %
                      (step+1, num_batches, np.exp(costs / (step+1)),time.time() - start_time))
                start_time = time.time()
            elif verbose and step % (num_batches // 10) == 0:
                print("batch %d of %d; perplexity: %.3f; time elapsed: %.3f s" %
                      (step+1, num_batches, np.exp(costs / (step+1)),time.time() - start_time))
                start_time = time.time()
        
        return np.exp(costs/num_batches)

    def predict(self, session, data, labels=None):
        """ Make predictions from provided model. If labels is not none, calculate loss. """

        losses,results = [],[[] for i in range(self.config.num_tasks)]
        for step, (x,y) in enumerate(self.enhancer_iterator(data, labels, 
                                                            self.config.batch_size,
                                                            self.config.num_steps)):
            if y is not None:
                a = session.run([ self.cost ]+[ pred for pred in self.predictions ],
                                { self.input_data: x,
                                  self.targets: y,
                                  self.dropout: 1.0,
                                  self.initial_state: self.initial_state.eval()})
                cost = a[0]
                preds = a[1:]
                losses.append(cost)
            else:
                preds = session.run([ pred for pred in self.predictions ],
                                    { self.input_data: x,
                                      self.dropout: 1.0,
                                      self.initial_state: self.initial_state.eval()})
            all_preds = [np.argmax(pred,1) for pred in preds]
            for task in range(config.num_tasks):
                results[task].extend(all_preds[task])
        return np.exp(np.mean(losses)), np.array(results).T

    def enhancer_iterator(self, data, labels, batch_size, num_steps):
        """
        Generate batch-size pointers on raw sequence data for minibatch iteration
        
        Args:
          data: length N array of sequences (nucleotides A C T G)
          labels: length N-by-num_tasks array of labels (integers) for each sequence
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
        mdata = np.array([seq_to_ints(i) for i in data], dtype=np.int32)
        num_batches = len(mdata) // batch_size
        
        # data will have batch_len elements, each of size batch_size
        # ASSUME FIXED SEQUENCE LENGTHS OFF 1000 FOR NOW (5/20/16)
        # Just grab middle self.config.num_steps nucleotides
        a = int(len(mdata[0,:])/2-self.config.num_steps/2)
        b = int(len(mdata[0,:])/2+self.config.num_steps/2)
        for i in range(num_batches):
            x = mdata[batch_size*i:batch_size*(i+1),a:b]
            if labels is not None:
                y = labels[batch_size*i:batch_size*(i+1),:]
            else:
                y = None
            yield(x,y)
        
def shuffle_in_unison_inplace(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def compute_error_rates(a,b):
    # 'a' are true labels, 'b' are estimated labels
    err_rates = []
    N,num_tasks = np.shape(b)
    for i in range(num_tasks):
        err_rates.append(sum(a[0:N,i]!=b[:,i])/float(N))
    return err_rates

def compute_f1_scores(a,b):
    # 'a' are true labels, 'b' are estimated labels
    f1_scores = []
    N,num_tasks = np.shape(b)
    for i in range(num_tasks):
        f1_scores.append(f1_score(a[0:N,i],b[:,i]))
    return f1_scores

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Need prefix for training/validation/testing datasets")
        sys.exit()

    train_data = np.loadtxt(sys.argv[1]+'.data.train.txt',dtype=str)
    train_labels = np.loadtxt(sys.argv[1]+'.labels.train.txt')
    valid_data = np.loadtxt(sys.argv[1]+'.data.valid.txt',dtype=str)
    valid_labels = np.loadtxt(sys.argv[1]+'.labels.valid.txt')
    test_data = np.loadtxt(sys.argv[1]+'.data.test.txt',dtype=str)
    test_labels = np.loadtxt(sys.argv[1]+'.labels.test.txt')
    config = DeepseaConfig()
    vocab = CharLevelDNAVocab()
    lr = config.learning_rate

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-0.01,0.01)
        best_val_loss = float('Inf')
        best_val_epoch = 0
        
        # Training
        print('setting up model..')
        with tf.variable_scope("model",reuse=None, initializer=initializer):
            m = EnhancerRNN(config=config,vocab=vocab)
        print('model set up')

        saver = tf.train.Saver()
        os.system("mkdir -p weights")
        tf.initialize_all_variables().run()

        for i in range(config.max_epoch):
            # Shuffle the data
            train_data,train_labels = shuffle_in_unison_inplace(train_data,train_labels)

            m.assign_lr(session, lr)
            print("="*80)
            print('Epoch %d with learning rate %.10f' %(i,lr))
            train_loss = m.run_epoch(session, train_data, train_labels)
            print("Train loss: %.3f" % (train_loss))
            valid_loss,valid_preds = m.predict(session, valid_data, valid_labels)
            valid_err = compute_error_rates(valid_labels,valid_preds)
            valid_f1 = compute_f1_scores(valid_labels,valid_preds)
            print("Valid loss: %.3f" % (valid_loss))
            print("Valid error rates: "+' '.join(['%.3f'%(j) for j in valid_err]))
            print("Valid f1 scores: "+' '.join(['%.3f'%(j) for j in valid_f1]))

            # Save training and validation losses 
            f = open('weights/validation_errors','a')
            f.write(str(train_loss)+'\t'+str(valid_loss)+'\t'+str(valid_err)+'\t'+str(valid_f1)+'\n')
            f.close()

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_val_epoch = i
                saver.save(session, './weights/weights.epoch'+str(i)+'.best')
                print('BEST EPOCH SO FAR: '+str(i))
                
            # If loss isn't getting better after a few iterations, decrease lr
#            if i-best_val_epoch > config.epochs_with_same_lr:
#                lr *= config.lr_decay
            # If loss isn't getting better after a lot of iterations, quit
            if i-best_val_epoch > config.early_stopping:
                break

            saver.save(session, './weights/weights.epoch'+str(i))

        saver.restore(session, './weights/weights.epoch'+str(best_val_epoch)+'.best')
        test_loss,test_preds = m.predict(session, test_data, test_labels)
        test_err = compute_error_rates(test_labels,test_preds)
        test_f1 = compute_f1_scores(test_labels,test_preds)
        print("="*80)
        print("Test loss: %.3f" % (test_loss))
        print("Test error rates: "+' '.join(['%.3f'%(j) for j in test_err]))
        print("Test f1 scores: "+' '.join(['%.3f'%(j) for j in test_f1]))
