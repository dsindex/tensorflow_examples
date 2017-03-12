#!./bin/env python

import tensorflow as tf
import numpy as np

# linear regression test
# y = W*x_data

xy_data = np.loadtxt('train_linear.txt', unpack=True, dtype='float32')
x_data = xy_data[0:-1]
y_data = xy_data[-1]

W = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))
y = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(2001):
	sess.run(train)
	if i % 20 == 0 :
		print i, sess.run(cost), sess.run(W)

