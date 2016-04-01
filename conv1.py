#!/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 28*28])
x_image = tf.reshape(x, [-1,28,28,1])     # 784 -> 28 x 28 x 1
y_ = tf.placeholder(tf.float32, [None, 10])

# first convolutional layer
W_conv1 = weight_variable([5, 5, 1, 32]) # 5 x 5 receptive field, 1 input channel, 32 feature maps
b_conv1 = bias_variable([32])            # 32 feature maps's bias
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 28 x 28 x 1 -> 0-padding -> 32 x 32 x 1 -> conv -> 28 x 28 x 32
h_pool1 = max_pool_2x2(h_conv1)                          # 28 x 28 x 32 -> pool -> 14 x 14 x 32 -> 0-padding -> 18 x 18 x 32
# 0-padding : http://cs.stackexchange.com/questions/49658/convolutional-neural-network-example-in-tensorflow

# second convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64]) # 5 x 5 receptive field, 32 input channel, 64 feature maps
b_conv2 = bias_variable([64])             # 64 feature maps's bias
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 18 x 18 x 32 -> 14 x 14 x 64
h_pool2 = max_pool_2x2(h_conv2)                          # 14 x 14 x 64 -> 7 x 7 x 64

# fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # 7 x 7 x 64 -> flat
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# readout layer(dropout, softmax)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# training
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for i in range(20000):
	batch_xs, batch_ys = mnist.train.next_batch(50)
	if i % 100 == 0:
		print "step : ", i, "training accuracy :", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})	
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

# inference
test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})	
print "test accuracy : ", test_accuracy

sess.close()
