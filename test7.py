#!/bin/env python

import tensorflow as tf
import numpy as np

d = tf.constant([ [ [0.5, 0.3, 0.1, -0.4, 0.1],
                    [0.34, 0.2, 0.4, 0.6, 0.1], 
                    [0.04, 0.3, 0.5, -0.9, 0.1],
                    [0.0, 0.0, 0.0, 0.0, 0.0] ],
                  [ [0.5, 0.3, 0.1, -0.4, 0.1],
                    [0.34, 0.2, 0.4, 0.6, 0.1], 
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0] ],
                  [ [0.5, 0.3, 0.1, -0.4, 0.1],
                    [0.34, 0.2, 0.4, 0.6, 0.1], 
                    [0.04, 0.3, 0.5, -0.9, 0.1],
                    [0.0, 0.0, 0.0, 0.0, 0.0] ] ], dtype=tf.float32)
                    
x = tf.constant([ [1, 1, 1, 0],
                  [1, 1, 0, 0], 
                  [1, 1, 1, 0] ], dtype=tf.float32)

y = tf.expand_dims(x, -1)
dim = 5
z = tf.tile(y, [1, 1, dim])

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(y)
	print sess.run(z)
    # element-wise
	print sess.run(d*z)
