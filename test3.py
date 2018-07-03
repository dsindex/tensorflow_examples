#!/bin/env python

import tensorflow as tf

x = tf.constant([1, -4])
y = tf.constant([ [ [2, 5],
                    [3, 4], 
                    [5, 7] ],
                  [ [1, -1],
                    [0, 4],
                    [9, 3] ] ])

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print sess.run(tf.abs(x))
	print sess.run(tf.reduce_max(y, reduction_indices=0))
	print sess.run(tf.reduce_max(y, reduction_indices=1))
	print sess.run(tf.reduce_max(y, reduction_indices=2))
	print sess.run(tf.sign(tf.reduce_max(y, reduction_indices=2)))
	print sess.run(tf.reduce_sum(tf.sign(tf.reduce_max(y, reduction_indices=2)),reduction_indices=1))
	print sess.run(tf.cast(tf.reduce_sum(tf.sign(tf.reduce_max(y, reduction_indices=2)),reduction_indices=1),tf.int32))

	print sess.run(tf.reduce_sum(y, reduction_indices=1))
	print sess.run(tf.reduce_sum(y, reduction_indices=0))

