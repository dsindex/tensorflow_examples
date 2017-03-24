#!./bin/env python

# simple character based RNN 
# ex) input 'hello worl', output 'ello world'

import tensorflow as tf
import numpy as np

def one_hot(i, vocab_size) :
	return [ 1 if j == i else 0 for j in xrange(vocab_size) ]

def build_dictionary(sentence) :
	idx2char = []
	for c in sentence :
		if c not in idx2char : idx2char.append(c)
	char2idx = {w: i for i, w in enumerate(idx2char)} # char to id
	return idx2char, char2idx

def build_x_data(sentence, sequence_length, char2idx) :
	x_data = sentence[0:sequence_length]                   # ['h','e','l','l','o',' ','w','o','r','l']
	vocab_size = len(char2idx)
	x_data = [char2idx[c] for c in x_data]         # [0, 1, 2, 2,...]
	x_data = [[one_hot(i, vocab_size) for i in x_data]]
	x_data = np.array(x_data, dtype='f')           # 1 x 10 x 8 = [None, sequence_length, input_dim]
	return x_data
	'''
	[[[ 1.  0.  0.  0.  0.  0.  0.  0.]   # 'h'
 	  [ 0.  1.  0.  0.  0.  0.  0.  0.]   # 'e'
	  [ 0.  0.  1.  0.  0.  0.  0.  0.]   # 'l'
	  [ 0.  0.  1.  0.  0.  0.  0.  0.]   # 'l'
	  [ 0.  0.  0.  1.  0.  0.  0.  0.]   # 'o'
	  [ 0.  0.  0.  0.  1.  0.  0.  0.]   # ' '
	  [ 0.  0.  0.  0.  0.  1.  0.  0.]   # 'w'
	  [ 0.  0.  0.  1.  0.  0.  0.  0.]   # 'o'
	  [ 0.  0.  0.  0.  0.  0.  1.  0.]   # 'r'
	  [ 0.  0.  1.  0.  0.  0.  0.  0.]]] # 'l'
	'''

def build_y_data(sentence, sequence_length, char2idx) :
	y_data = [[char2idx[c] for c in sentence]]   # 1 x 10 = [None, sequence_length]
	return y_data
	'''
	[[1, 2, 2, 3, 4, 5, 3, 6, 2, 7]]
	'''

def rnn_model(hidden_sizie, batch_size, X) :
	cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
	initial_state = cell.zero_state(batch_size, tf.float32)
	outputs, _states = tf.nn.dynamic_rnn(cell=cell, inputs=X, initial_state=initial_state, dtype=tf.float32)
	print outputs
	return outputs

sentence = 'hello world'                           # sentence = 'this is a simple RNN'
sentence_size = len(sentence)
print 'sentence = ' + sentence
print 'sentence size = %d' % sentence_size

# build dictionary
idx2char, char2idx = build_dictionary(sentence)
vocab_size = len(char2idx)
print 'vocab size = %d' % vocab_size               # vocab size : 8

# config
input_dim = vocab_size                             # one-hot size
hidden_size = vocab_size                           # 8, output from LSTM, directly predict one-hot
sequence_length = sentence_size - 1                # 10
batch_size = 1                                     # 1 sentence

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])               # Y label

outputs = rnn_model(hidden_size, batch_size, X)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs, axis=2)


# build a x_data
x_data = build_x_data(sentence, sequence_length, char2idx)
print x_data
y_data = build_y_data(sentence[1:], sequence_length, char2idx)
print y_data


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(200):
		l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
		result = sess.run(prediction, feed_dict={X: x_data})
		print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

		# print char using dic
		result_str = [idx2char[c] for c in np.squeeze(result)]
		print("\tPrediction str: ", ''.join(result_str))
