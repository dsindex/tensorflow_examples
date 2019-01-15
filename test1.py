#!/bin/env python

from __future__ import print_function
import tensorflow as tf

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print('state : {}, one : {}'.format(sess.run(state), sess.run(one)))
    print('new_value : {}, update : {}'.format(sess.run(new_value), sess.run(update)))
    print('state : {}'.format(sess.run(state)))
    for _ in range(3):
        sess.run(update)
        print('state : {}'.format(sess.run(state)))

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)
with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print('[mul, intermed] = [%s, %s]' % (result[0], result[1]))

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    ret = sess.run([output], feed_dict={input1:7., input2:2.})
    print(type(ret))
    print(ret)

import numpy as np
x = tf.placeholder(tf.float32, shape=(1024))
y = tf.multiply(x, x)
with tf.Session() as sess:
    rand_array = np.random.rand(1024)
    ret = sess.run(y, feed_dict={x: rand_array})
    print(type(ret))
    print(ret)

x = tf.placeholder(tf.float32, shape=(1024, 512))
x_t = tf.transpose(x)
y = tf.matmul(x, x_t)
with tf.Session() as sess:
    rand_array = np.random.rand(1024, 512)
    ret = sess.run(y, feed_dict={x: rand_array})
    print(y)
    print(type(ret))
    print(ret)

x = tf.placeholder(tf.float32, shape=(1024))
y = tf.placeholder(tf.float32, shape=(1024))
s = tf.reduce_sum(tf.multiply(x, y))
with tf.Session() as sess:
    rand_array1 = np.random.rand(1024)
    rand_array2 = np.random.rand(1024)
    ret = sess.run(s, feed_dict={x: rand_array1, y: rand_array2})
    print(type(ret))
    print(ret)

x = tf.placeholder(tf.float32, shape=(1024))
print(x)
x_mat = tf.reshape(x, [1, -1])
print(x_mat)
y = tf.placeholder(tf.float32, shape=(1024))
print(y)
y_mat = tf.reshape(y, [-1, 1])
print(y_mat)
s = tf.matmul(x_mat, y_mat)
print(s)
scalar = tf.reshape(s, [])
with tf.Session() as sess:
    rand_array1 = np.random.rand(1024)
    rand_array2 = np.random.rand(1024)
    ret = sess.run(scalar, feed_dict={x: rand_array1, y: rand_array2})
    print(type(ret))
    print(ret)


