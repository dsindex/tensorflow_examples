#!./bin/env python

# simple character based RNN 
# ex) input 'hello worl', output 'ello world'

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

def one_hot(x_data, padding=0) :
	a = np.array(x_data, dtype=int)
	b = np.zeros((a.size, a.max()+1+padding))
	b[np.arange(a.size),a] = 1
	return b

def build_dictionary(sentence) :
	char_rdic = []
	for c in sentence :
		if c not in char_rdic : char_rdic.append(c)
	char_dic = {w: i for i, w in enumerate(char_rdic)} # char to id
	return char_rdic, char_dic

def build_x_data(sentence, n_steps, char_dic) :
	x_data = sentence[0:n_steps]                   # ['h','e','l','l','o',' ','w','o','r','l']
	vocab_size = len(char_dic)
	padding = vocab_size - len(set(x_data))
	x_data = [ [char_dic[c] for c in x_data] ]     # [ [0, 1, 2, 2,...] ]
	x_data = one_hot(x_data, padding)
	x_data = np.array(x_data, dtype='f')           # 10 x 8 matrix
	return x_data
	'''
	[[ 1.  0.  0.  0.  0.  0.  0.  0.]  # 'h'
	 [ 0.  1.  0.  0.  0.  0.  0.  0.]  # 'e'
	 [ 0.  0.  1.  0.  0.  0.  0.  0.]  # 'l'
	 [ 0.  0.  1.  0.  0.  0.  0.  0.]  # 'l'
	 [ 0.  0.  0.  1.  0.  0.  0.  0.]  # 'o'
	 [ 0.  0.  0.  0.  1.  0.  0.  0.]  # ' '
	 [ 0.  0.  0.  0.  0.  1.  0.  0.]  # 'w'
	 [ 0.  0.  0.  1.  0.  0.  0.  0.]  # 'o'
	 [ 0.  0.  0.  0.  0.  0.  1.  0.]  # 'r'
	 [ 0.  0.  1.  0.  0.  0.  0.  0.]] # 'l'
	'''

def build_targets(sentence, n_steps, char_dic) :
	sample = [char_dic[c] for c in sentence]
	targets = tf.reshape(sample[1:n_steps+1], [-1]) # (10,) tensor
	return targets
	'''
	ex) [1, 2, 2, 3, 4, 5, 3, 6, 2, 7]
	    (10,) expected output sequence
	'''

def rnn_model(rnn_size, n_steps, x_data, batch_size) :
	'''
	outputs  (1,8) (1,8) ... (1,8) (1,8)
	cell     BasicRNNCell(8)
	state    (1,8)
	X_split  (1,8) (1,8) ... (1,8) (1,8)
	'''
	cell = rnn_cell.BasicRNNCell(rnn_size)
	print 'rnn_size = %d' % rnn_size
	istate = tf.zeros([batch_size, cell.state_size])  # (1,8)
	print 'istate : '
	print istate
	X_split = tf.split(0, n_steps, x_data)            # (10,8) -> (1,8),(1,8),...,(1,8),(1,8)
	print 'X_split : '
	print X_split
	outputs, state = rnn.rnn(cell, X_split, istate)
	print 'outputs : '
	print outputs
	return outputs, state

sentence = 'hello world'                           # sentence = 'this is a simple RNN'
sentence_size = len(sentence)
print 'sentence = ' + sentence
print 'sentence size = %d' % sentence_size

# build dictionary
char_rdic, char_dic = build_dictionary(sentence)
vocab_size = len(char_dic)
print 'vocab size = %d' % vocab_size               # vocab size : 8

# config
rnn_size = vocab_size                              # output layer dimension(decoder symbols) : 8
n_steps = sentence_size - 1                        # input layer time step : 10
batch_size = 1                                     # 1 sample

# build a x_data
x_data = build_x_data(sentence, n_steps, char_dic)

# RNN model
outputs, state = rnn_model(rnn_size, n_steps, x_data, batch_size)

# build a targets
# targets : list of 1D batch-sized int32 Tensors of the same length as logits
targets = build_targets(sentence, n_steps, char_dic)

# logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols]
#         combine splits into (10,8)
# ex) [ [0.1,0.3,0.3,0.3,..], [0.2,0.3,0.4,0.1,..],...,[0.5,0.3,0.1,0.1,..], [0.7,0.2,0.1,0.0,..] ]
#     (10, 8) predicted output sequence
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])

# weights: list of 1D batch-sized float-Tensors of the same length as logits
# ex) [1, 1, ..., 1, 1]
#     (10,) weights
weights = tf.ones([n_steps * batch_size]) 

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
