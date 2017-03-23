#!/bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 28*28])
y_ = tf.placeholder(tf.float32, [None, 10])

# hidden layer
W_h1 = weight_variable([28*28, 512])
b_h1 = bias_variable([512])
h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_h1)

# output layer
W_out = weight_variable([512, 10])
b_out = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h1, W_out) + b_out)

# training
cost = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
NUM_THREADS = 5
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,inter_op_parallelism_threads=NUM_THREADS,log_device_placement=False))
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver() # save all variables
import os
checkpoint_dir = './train_logs/'
if not os.path.exists(checkpoint_dir) :
    os.mkdir(checkpoint_dir, 0755)
checkpoint_file = 'mlp.ckpt'
for i in range(10000) :
        batch_xs, batch_ys = mnist.train.next_batch(50)
        if i % 100 == 0 :
                print "step : ", i
                p, a = sess.run([correct_prediction, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
                print "prediction for batch: "
                print p
                print "accuracy for batch : ", a

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# save model
saver.save(sess, checkpoint_dir + checkpoint_file)
print "model saved in ", checkpoint_dir + checkpoint_file

sess.close()
