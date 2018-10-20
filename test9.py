#!/bin/env python

import tensorflow as tf
import numpy as np

count=0
def func(x):
    global count
    count += 1
    return x

n = tf.placeholder(tf.int32)
x = tf.range(n)
c = lambda i, x: tf.less(i, n)
b = lambda i, x: (i+1, func(x))
i, out = tf.while_loop(c, b, (tf.constant(0), x))

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print sess.run([i, out], feed_dict={n:10})
    print 'count = ', count
