#!/bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np

x = tf.constant([ [0, 0, -1000, -1000],
                  [0, 0, 0, 0], 
                  [0, 0, 0, -1000] ] )

y = tf.expand_dims(x, 0)

z = tf.transpose(y, [1, 0, 2])

k = tf.tile(z, [1, 2, 1])
print(k)

w = tf.expand_dims(k, 1)
print(w)

j = tf.transpose(w, [0, 2, 1, 3])
print(j)

u = tf.tile(j, [1, 1, 4, 1])
print(u)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(y))
    print(sess.run(z))
    print(sess.run(k))
    print(sess.run(w))
    print(sess.run(j))
    print(sess.run(u))
