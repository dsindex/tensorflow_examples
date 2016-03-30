#!./bin/env python

import tensorflow as tf

# linear regression test
# y = W1*x1_data + W2*x2_data + b

x1_data = [1., 0., 3., 0., 5.]
x2_data = [0., 2., 0., 4., 0.]
y_data  = [1., 2., 3., 4., 5.]

W1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
y = W1*x1_data + W2*x2_data + b

cost = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(2001):
	sess.run(train)
	if i % 20 == 0 :
		print i, sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b)

