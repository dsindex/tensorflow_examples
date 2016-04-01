#!/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

print '[training]'
# Dataset loading
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Session
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

# Learning
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	print i, sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print '[inference]'
# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Result should be approximately 91%.
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})

sess.close()
