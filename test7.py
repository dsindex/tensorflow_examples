#!/bin/env python

import tensorflow as tf
import numpy as np

x = tf.constant([ [1, 1, 1, 0],
                  [1, 1, 0, 0], 
                  [1, 1, 1, 0] ] )

y = tf.expand_dims(x, -1)
dim = 5
z = tf.tile(y, [1, 1, dim])

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(y)
	print sess.run(z)
