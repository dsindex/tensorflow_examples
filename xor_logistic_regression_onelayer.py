#!./bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np

# https://aimatters.wordpress.com/2016/01/16/solving-xor-with-a-neural-network-in-tensorflow/
# https://copycode.tistory.com/195


x_ = tf.placeholder(tf.float32, shape=[4,3], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4,1], name="y-input")
Theta1 = tf.Variable(tf.random_uniform([3,2], -1, 1), name="Theta1")
Bias1 = tf.Variable(tf.zeros([2]), name="Bias1")
Hypothesis = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
cost = tf.reduce_mean(( (y_ * tf.log(Hypothesis)) + ((1 - y_) * tf.log(1.0 - Hypothesis)) ) * -1)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

predicted = tf.cast(Hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_), dtype=tf.float32))  


# x1, x2, x1*x2
XOR_X = [[0,0,0],
         [0,1,0],
         [1,0,0],
         [1,1,1]]
XOR_Y = [[0],
         [1],
         [1],
         [0]]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(100000):
    sess.run(train_step, feed_dict={x_: XOR_X, y_: XOR_Y})
    if i % 1000 == 0:
        print('Epoch ', i)
        h, p, a = sess.run([Hypothesis, predicted, accuracy], feed_dict={x_: XOR_X, y_: XOR_Y})
        print('Hypothesis ', h)
        print('Predicted ', p)
        print('Accuracy ', a)
        print('Theta1 ', sess.run(Theta1))
        print('Bias1 ', sess.run(Bias1))
        print('cost ', sess.run(cost, feed_dict={x_: XOR_X, y_: XOR_Y})) 

sess.close()

