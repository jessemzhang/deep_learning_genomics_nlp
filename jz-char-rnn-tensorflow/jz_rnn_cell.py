import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops

# Slightly modifying RNN cell classes so that activation function can be changed

class jzRNNCell(rnn_cell.BasicRNNCell):
  def __init__(self, num_units, input_size=None, activation=tf.tanh):
    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size
    self._activation = activation
  def __call__(self, inputs, state, scope=None):
    """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"            
      output = self._activation(rnn_cell.linear([inputs, state], self._num_units, True))
    return output, output

class jzGRUCell(rnn_cell.GRUCell):
  def __init__(self, num_units, input_size=None, activation=tf.tanh):
    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size
    self._activation = activation
  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope(scope or type(self).__name__):  # "GRUCell"                 
      with vs.variable_scope("Gates"):  # Reset gate and update gate.                  
        # We start with bias of 1.0 to not reset and not update.                       
        r, u = array_ops.split(1, 2, rnn_cell.linear([inputs, state],
                                            2 * self._num_units, True, 1.0))
        r, u = tf.sigmoid(r), tf.sigmoid(u)
      with vs.variable_scope("Candidate"):
        c = self._activation(rnn_cell.linear([inputs, r * state], self._num_units, True))
      new_h = u * state + (1 - u) * c
    return new_h, new_h

class jzLSTMCell(rnn_cell.BasicLSTMCell):
  def __init__(self, num_units, forget_bias=1.0, input_size=None, activation=tf.tanh):
    self._num_units = num_units
    self._input_size = num_units if input_size is None else input_size
    self._forget_bias = forget_bias
    self._activation = activation
  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"           
      # Parameters of gates are concatenated into one multiply for efficiency.         
      c, h = array_ops.split(1, 2, state)
      concat = rnn_cell.linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate                
      i, j, f, o = array_ops.split(1, 4, concat)

      new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * self._activation(j)
      new_h = self._activation(new_c) * tf.sigmoid(o)

      return new_h, array_ops.concat(1, [new_c, new_h])
