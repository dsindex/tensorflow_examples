#!/bin/env python

import tensorflow as tf
import numpy as np

x = tf.constant([ [0, 0, -1000, -1000],
                  [0, 0, 0, 0], 
                  [0, 0, 0, -1000] ] )

y = tf.expand_dims(x, 0)
z = tf.transpose(y, [1, 0, 2])
w = tf.expand_dims(z, 1)
print w
u = tf.tile(w, [1, 1, 4, 1])
print u

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(y)
	print sess.run(z)
	print sess.run(w)
	print sess.run(u)
