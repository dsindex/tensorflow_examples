#!/bin/env python

import tensorflow as tf
import numpy as np

x = tf.constant([ [0, 0, -1000, -1000],
                  [0, 0, 0, 0], 
                  [0, 0, 0, -1000] ] )

y = tf.expand_dims(x, -1)
z = tf.tile(y, [1, 1, 5])

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(z)
