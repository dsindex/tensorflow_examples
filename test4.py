#!/bin/env python

import tensorflow as tf
import numpy as np

y = tf.constant([ [2, 5, 7, 8],
                  [3, 4, 10, 2], 
                  [5, 7, 3, -2] ] )

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(tf.reduce_max(y, reduction_indices=0))
	# [ 5  7 10  8]
	print sess.run(tf.reduce_max(y, reduction_indices=1))
	# [ 8 10  7]
