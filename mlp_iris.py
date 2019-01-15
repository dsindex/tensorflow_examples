#!./bin/env python

from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np

def one_hot(y_data) :
    a = np.array(y_data, dtype=int)
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

xy_data = np.loadtxt('train_iris.txt', unpack=True, dtype='float32')

x_data = xy_data[1:]
train_size = len(x_data[0])
# add x0 for bias
x0 = np.array([[1]*train_size])
x_data = np.concatenate((x0, x_data), axis=0) # 5 x None
x_data = np.transpose(x_data)                 # None x 5
'''
[ [1    2    14    33    50],
  [1    24    56    31    67],
  [1    23    51    31    69],
  .... ]
'''
print(x_data)

y_data = xy_data[0]                # 1 x None
'''
[ [0 1 1 0 1 2 .... ] ]
'''
y_data = one_hot(y_data)           # None x 3
'''
[ [1 0 0],
  [0 1 0],
  [0 1 0],
  ... ]
'''
print(y_data)

X = tf.placeholder("float", [None, 5]) # row : infinity, col : 5 for x
Y = tf.placeholder("float", [None, 3]) # row : infinity, col : 3 for y target class which is encoded in one-hot representation

# hidden layer
W_h1 = weight_variable([5, 13])
h1 = tf.nn.sigmoid(tf.matmul(X, W_h1))

# output layer
W_out = weight_variable([13, 3])
b_out = bias_variable([3])
y = tf.nn.softmax(tf.matmul(h1, W_out) + b_out) # None x 3

# training
# cross entropy cost
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(y), reduction_indices=1))

learning_rate = tf.Variable(0.001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(20000):
    if i % 100 == 0 :
        print("step : ", i)
        print("cost : ", sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        print("training accuracy :", sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
    sess.run(train, feed_dict={X:x_data, Y:y_data})

sess.close()
