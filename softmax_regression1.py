#!./bin/env python

import tensorflow as tf
import numpy as np

# softmax regression test

'''
# x0 x1 x2 A B C
1 2 1 0 0 1
1 3 2 0 0 1
1 3 4 0 0 1
1 5 5 0 1 0
1 7 5 0 1 0
1 2 5 0 1 0
...
'''
xy_data = np.loadtxt('train_softmax.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy_data[0:3]) # [ [1,1,1,1,1],    <-- 3 x None matrix
	                                #   [2,3,3,5,7,2], 
	                                #   [1,2,4,5,5,5] ]
	                                # transpose T
	                                # [ [1,2,1],        <-- None x 3 matrix
	                                #   [1,3,2],
	                                #   [1,3,4],
	                                #   [1,5,5],
	                                #   [1,7,5],
	                                #   [1,2,5] ]
y_data = np.transpose(xy_data[3:])  # [ [0,0,0,0,0],    <-- 3 x None matrix
	                                #   [0,0,0,1,1,1], 
	                                #   [1,1,1,0,0,0] ]
	                                # transpose T       <-- None x 3 matrix
	                                # [ [0,0,1],
	                                #   [0,0,1],
	                                #   [0,0,1],
	                                #   [0,1,0],
	                                #   [0,1,0],
	                                #   [0,1,0] ]

X = tf.placeholder("float", [None, 3]) # row : infinity, col : 3 for x0/x1/x2
Y = tf.placeholder("float", [None, 3]) # row : infinity, col : 3 for A/B/C target class which is encoded in one-hot representation

# output layer
W = tf.Variable(tf.zeros([3,3])) # row : 3 dimensions for x, col : 3 dimensions for y
y = tf.nn.softmax(tf.matmul(X, W)) # softmax, (None x 3) * ( 3 x 3 )

# training
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

# inference
p = sess.run(y, feed_dict={X:[[1,2,1]]}) # 0 0 1 -> C, index 2
print p, sess.run(tf.arg_max(p, 1))

p = sess.run(y, feed_dict={X:[[1,5,5]]}) # 0 1 0 -> B, index 1
print p, sess.run(tf.arg_max(p, 1))

p = sess.run(y, feed_dict={X:[[1,2,1], [1,5,5]]})
print p, sess.run(tf.arg_max(p, 1))

sess.close()
