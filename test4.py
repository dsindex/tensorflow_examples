#!/bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np

y = tf.constant([ [2, 5, 7, 8],
                  [3, 4, 10, 2], 
                  [5, 7, 3, -2] ] )

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(tf.reduce_min(y, axis=-2, keep_dims=True)))
    # [[ 2  4  3 -2]]
    print(sess.run(tf.reduce_min(y, axis=-1)))
    # [ 2  2 -2]
