import os,itertools,sys
import multiprocessing as mp

text = sys.argv[1]
textdir = 'data/'+text
sd = 'save_'+text
os.system('mkdir -p '+sd)
os.system('mkdir -p '+sd+'/train_losses')

# combinations of hyperparameters to test
models = ['rnn','lstm','gru']
layers = [2,3]
seq_lengths = [50,100,500,1000]
learning_rates = [0.0001]

def train_char_rnn(inputs):
    m,nl,sl,lr = inputs
    os.system('python train.py --data_dir ' + textdir + ' --save_dir ' + sd + ' --num_layers ' + str(nl) + ' --model ' + m + ' --seq_length ' + str(sl) + ' --num_epochs 1 ' + ' --learning_rate ' + str(lr))


allinputs = list(itertools.product(models,layers,seq_lengths,learning_rates))
pool=mp.Pool(processes=8)
pool.map(train_char_rnn,allinputs)
