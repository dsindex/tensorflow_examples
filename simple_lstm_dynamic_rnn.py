#!/bin/env python

import tensorflow as tf
import numpy as np

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def build_dictionary(sentences) :
	char_rdic = []
	for sentence in sentences :
		for c in sentence :
			if c not in char_rdic : char_rdic.append(c)
	char_dic = {w: i for i, w in enumerate(char_rdic)} # char to id
	return char_rdic, char_dic

def one_hot(i, vocab_size) :
	return [ 1 if j == i else 0 for j in xrange(vocab_size) ]

def next_batch(sentences, begin, batch_size, n_steps, char_dic) :
	batch_xs = []
	batch_ys = []
	batch_xls = []
	count = 0
	for sentence in sentences[begin:] :
		x_data = sentence[0:n_steps]
		batch_xls.append(len(x_data))
		vocab_size = len(char_dic)
		x_data = [char_dic[c] for c in x_data]
		x_data = [one_hot(i, vocab_size) for i in x_data]
		batch_xs.append(x_data)
		y_data = [char_dic[c] for c in sentence[1:n_steps+1]]
		batch_ys.append(y_data)
		count += 1
		if count == batch_size : break
	batch_xs = np.array(batch_xs, dtype='f')
	batch_ys = np.array(batch_ys, dtype='int32')
	batch_xls = np.array(batch_xls, dtype='int32')
	return batch_xs, batch_ys, batch_xls, begin+count


sentences = ['abcdefg',  # train
	     'hijklmn',
	     'opqrstu',
	     'vwxyz**',
	     'abcdefg',  # test
	     'opqrstu']

# config
learning_rate = 0.01
training_iters = 500

n_steps = len(sentences[0]) - 1 # time stpes
char_rdic, char_dic = build_dictionary(sentences)
n_input = len(char_dic)         # input dimension, vocab size
n_hidden = 8                    # hidden state size = lstm_size
n_classes = len(char_dic)       # output classes,  vocab size

x = tf.placeholder("float", [None, n_steps, n_input])
y_ = tf.placeholder("int32", [None, n_steps])

# LSTM layer
weights = {
	'out' : weight_variable([n_hidden, n_classes])
}
biases = {
	'out': bias_variable([n_classes])
}

def RNN(_x, _x_lengths, _weights, _biases):
	# input _X shape: (batch_size, n_steps, n_input)
	# Define a lstm cell with tensorflow
	lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
	# Get lstm cell output
	# to save computation time and ensure correctness, provide length list
	outputs, states = tf.nn.dynamic_rnn(
				cell=lstm_cell,
				dtype=tf.float32,
				sequence_length=_x_lengths,
				inputs=_x)

	# outputs == (?,n_steps,n_hidden)
	# _weights['out'] = (n_hidden, n_classes)
	outputs = tf.reshape(outputs, [-1, n_hidden])
	final_outputs = tf.matmul(outputs, _weights['out']) + _biases['out'] # (?, n_classes)
	return final_outputs

# training
batch_size = 4
x_lengths = tf.placeholder("int32", [batch_size])
y = RNN(x, x_lengths, weights, biases)

logits = tf.reshape(y, [-1, n_classes])
targets = y_
seq_weights = tf.ones([n_steps * batch_size])
loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [seq_weights])
cost = tf.reduce_sum(loss) / batch_size 
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

NUM_THREADS = 1
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,inter_op_parallelism_threads=NUM_THREADS,log_device_placement=False))
init = tf.initialize_all_variables()
sess.run(init)

begin = 0
batch_xs, batch_ys, batch_xls, begin = next_batch(sentences, begin, batch_size, n_steps, char_dic)
print 'batch_xs.shape : ' + str(batch_xs.shape)
print 'batch_xs : '
print batch_xs
print 'batch_ys.shape : ' + str(batch_ys.shape)
print 'batch_ys : '
print batch_ys
step = 1
while step < training_iters :
	c_istate = np.zeros((batch_size, 2*n_hidden))
	feed={x: batch_xs, y_: batch_ys, x_lengths: batch_xls}
	sess.run(optimizer, feed_dict=feed)
	if step % 10 == 0 : 
		print 'step : %s' % step + ',' + 'cost : %s' % sess.run(cost, feed_dict=feed)
		results = sess.run(tf.arg_max(logits, 1), feed_dict=feed)
		results = np.reshape(results, (batch_size,-1))
		for result in results :
			print result, [char_rdic[t] for t in result]
	step += 1

import sys
sys.exit(0)

# inference
batch_size = 2  ### FIXME not equal to x_lengths dimention, how to solve it? split training and inference
batch_xs, batch_ys, batch_xls, begin = next_batch(sentences, begin, batch_size, n_steps, char_dic)
c_istate = np.zeros((batch_size, 2*n_hidden))
feed={x: batch_xs, y_: batch_ys, x_lengths: batch_xls}
result = sess.run(tf.arg_max(logits, 1), feed_dict=feed)
print result, [char_rdic[t] for t in result]


