#!./bin/env python

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

def one_hot(x_data, padding=0) :
	a = np.array(x_data, dtype=int)
	b = np.zeros((a.size, a.max()+1+padding))
	b[np.arange(a.size),a] = 1
	return b

sentence = 'hello'

# build dictionary
char_rdic = []
for c in sentence :
	if c not in char_rdic : char_rdic.append(c)
char_dic = {w: i for i, w in enumerate(char_rdic)} # char to id

# build x_data
x_data = sentence[:-1]                             # ['h','e','l','l']
x_data_size = len(x_data)
count=0
for c in char_rdic :
	if c in x_data : count += 1
padding = x_data_size - count
x_data = [ [char_dic[c] for c in x_data] ]         # [ [0, 1, 2, 2] ]
x_data = one_hot(x_data, padding)
x_data = np.array(x_data, dtype='f')
'''
[ [1,0,0,0],   # h
  [0,1,0,0],   # e
  [0,0,1,0],   # l
  [0,0,1,0] ], # l
'''
print x_data

sample = [char_dic[c] for c in sentence]
print sample

# config
char_vocab_size = len(char_dic) 
rnn_size = char_vocab_size     # output layer dimension 4
time_step_size = 4             # input layer time step
batch_size = 1                 # 1 sample

# RNN model
rnn_cell = rnn_cell.BasicRNNCell(rnn_size)
istate = tf.zeros([batch_size, rnn_cell.state_size])
X_split = tf.split(0, time_step_size, x_data) 
outputs, state = rnn.rnn(rnn_cell, X_split, istate)

# logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols]
# ex) [ [0.1,0.3,0.3,0.3], [0.2,0.3,0.4,0.1], [0.5,0.3,0.1,0.1], [0.7,0.2,0.1,0.0] ]
#     4 x 4 predicted output sequence
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size]) 
# targets: list of 1D batch-sized int32 Tensors of the same length as logits
# ex) [1, 2, 2, 3]
#     1 x 4 expected output sequence
targets = tf.reshape(sample[1:], [-1])
# weights: list of 1D batch-sized float-Tensors of the same length as logits
# ex) [1, 1, 1, 1]
weights = tf.ones([time_step_size * batch_size]) 

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size 
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(100) :
	sess.run(train_op) 
	result = sess.run(tf.arg_max(logits, 1)) 
	print result, [char_rdic[t] for t in result] 

sess.close()
