#!./bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np

# logistic regression test
# y = 1 / ( 1 + e^(-WX) )

xy_data = np.loadtxt('train_logistic.txt', unpack=True, dtype='float32')
x_data = xy_data[0:-1]
y_data = xy_data[-1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# output layer
W = tf.Variable(tf.random_uniform([1,len(x_data)], -1.0, 1.0))
g = tf.matmul(W, X)
y = tf.div(1., 1.+tf.exp(-g))

# training
# cross entropy cost =  -(1/m) * { y_data*log(y) + (1-y_data)*log(1-y) }
cost = -tf.reduce_mean(Y*tf.log(y) + (1-Y)*tf.log(1-y))
# why we'd better to use cross entropy cost function rather than quardratic cost function?
# http://neuralnetworksanddeeplearning.com/chap3.html 'learning slow problem'

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if i % 20 == 0 :
        print(i, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

# inference
print(sess.run(y, feed_dict={X:[[1], [2], [3]]}))
print(sess.run(y, feed_dict={X:[[1], [5], [5]]}))
print(sess.run(y, feed_dict={X:[[1,1], [4,3], [3,5]]}))

sess.close()

