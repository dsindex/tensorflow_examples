#!/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# output layer
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# training
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.global_variables_initializer()
NUM_THREADS = 5
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,inter_op_parallelism_threads=NUM_THREADS,log_device_placement=True))
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(20000):
	batch_xs, batch_ys = mnist.train.next_batch(50)
	if i % 100 == 0:
		print "step : ", i, "training accuracy :", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})	
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# inference
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

sess.close()
