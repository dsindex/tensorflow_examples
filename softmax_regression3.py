#!./bin/env python

import sys
import tensorflow as tf
import numpy as np

def one_hot(y_data) :
	a = np.array(y_data, dtype=int)
	b = np.zeros((a.size, a.max()+1))
	b[np.arange(a.size),a] = 1
	return b

print '[training]'
xy_data = np.loadtxt('train_iris.txt', unpack=True, dtype='float32')

x_data = xy_data[1:]
batch_size = len(x_data[0])
# add x0 for bias
x0 = np.array([[1]*batch_size])
x_data = np.concatenate((x0, x_data), axis=0) # 5 x None
x_data = np.transpose(x_data)                 # None x 5
'''
[ [1	2	14	33	50],
  [1	24	56	31	67],
  [1	23	51	31	69],
  .... ]
'''
print x_data

y_data = xy_data[0]                # 1 x None
'''
[ [0 1 1 0 1 2 .... ] ]
'''
y_data = one_hot(y_data)           # None x 3
'''
[ [1 0 0],
  [0 1 0],
  [0 1 0],
  ... ]
'''
print y_data

X = tf.placeholder("float", [None, 5]) # row : infinity, col : 5 for x
Y = tf.placeholder("float", [None, 3]) # row : infinity, col : 3 for y target class which is encoded in one-hot representation

W = tf.Variable(tf.zeros([5,3])) # row : 3 dimensions for x, col : 3 dimensions for y

y = tf.nn.softmax(tf.matmul(X, W)) # softmax, (None x 5) * ( 5 x 3 )

# cross entropy cost
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(y), reduction_indices=1))

learning_rate = tf.Variable(0.001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(2001):
	sess.run(train, feed_dict={X:x_data, Y:y_data})
	if i % 20 == 0 :
		print i, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

print '[inference]'
p = sess.run(y, feed_dict={X:[[1,2,14,33,50]]}) # 1 0 0 -> type 0
print p, sess.run(tf.arg_max(p, 1))

p = sess.run(y, feed_dict={X:[[1,24,56,31,67]]}) # 0 1 0 -> type 1
print p, sess.run(tf.arg_max(p, 1))

p = sess.run(y, feed_dict={X:[[1,2,14,33,50], [1,24,56,31,67]]})
print p, sess.run(tf.arg_max(p, 1))
