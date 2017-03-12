#!./bin/env python

# simple character based RNN 
# ex) input 'hello worl', output 'ello world'

import tensorflow as tf
import numpy as np

def one_hot(i, vocab_size) :
	return [ 1 if j == i else 0 for j in xrange(vocab_size) ]

def build_dictionary(sentence) :
	char_rdic = []
	for c in sentence :
		if c not in char_rdic : char_rdic.append(c)
	char_dic = {w: i for i, w in enumerate(char_rdic)} # char to id
	return char_rdic, char_dic

def build_x_data(sentence, n_steps, char_dic) :
	x_data = sentence[0:n_steps]                   # ['h','e','l','l','o',' ','w','o','r','l']
	vocab_size = len(char_dic)
	x_data = [char_dic[c] for c in x_data]         # [0, 1, 2, 2,...]
	x_data = [one_hot(i, vocab_size) for i in x_data]
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

def rnn_model(rnn_size, n_steps, x_data) :
	'''
	outputs  (1,8) (1,8) ... (1,8) (1,8)
	cell     BasicRNNCell(8)
	state    (1,8)
	X_split  (1,8) (1,8) ... (1,8) (1,8)
	'''
	#cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
	cell = tf.contrib.rnn.BasicRNNCell(rnn_size)
	print 'rnn_size = %d' % rnn_size
	istate = tf.zeros([1, cell.state_size])  # (1,8)
	print 'istate : '
	print istate
	#X_split = tf.split(0, n_steps, x_data)   # (10,8) -> (1,8),(1,8),...,(1,8),(1,8)
	X_split = tf.split(x_data, n_steps, 0)   # (10,8) -> (1,8),(1,8),...,(1,8),(1,8)
	print 'X_split : '
	print X_split
	#outputs, state = tf.nn.rnn(cell, X_split, istate)
	outputs, state = tf.nn.dynamic_rnn(cell=cell, inputs=X_split)
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
outputs, state = rnn_model(rnn_size, n_steps, x_data)

# build a targets
'''
# targets : list of 1D batch-sized int32 Tensors of the same length as logits
            ex) [1, 2, 2, 3, 4, 5, 3, 6, 2, 7]
            (10,) expected output sequence
'''
targets = build_targets(sentence, n_steps, char_dic)
print targets
'''
# logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols]
          combine (1, 8) splits into (10,8)
          ex) [ [0.1,0.3,0.3,0.3,..], [0.2,0.3,0.4,0.1,..],...,[0.5,0.3,0.1,0.1,..], [0.7,0.2,0.1,0.0,..] ]
          (10, 8) predicted output sequence
'''
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
print logits

'''
# weights: list of 1D batch-sized float-Tensors of the same length as logits
           ex) [1, 1, ..., 1, 1]
           (10,) weights
'''
weights = tf.ones([n_steps * batch_size]) 

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size 
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(100) :
	sess.run(train_op)
	result = sess.run(tf.arg_max(logits, 1)) 
	print result, [char_rdic[t] for t in result] 

sess.close()
