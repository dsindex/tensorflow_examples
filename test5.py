#!/bin/env python

import tensorflow as tf
import numpy as np

x = tf.constant([ [0, 0, -1000, -1000],
                  [0, 0, 0, 0], 
                  [0, 0, 0, -1000] ] )

y = tf.expand_dims(x, 0)
z = tf.expand_dims(y, 0)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(y)
	print sess.run(z)
