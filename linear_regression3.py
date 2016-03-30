#!./bin/env python

import tensorflow as tf

# linear regression test
# y = W*x_data

x_data = [[1., 1., 1., 1., 1.],
          [1., 0., 3., 0., 5.],
          [0., 2., 0., 4., 0.]]
y_data  = [1., 2., 3., 4., 5.]

W = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))
y = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(2001):
	sess.run(train)
	if i % 20 == 0 :
		print i, sess.run(cost), sess.run(W)

