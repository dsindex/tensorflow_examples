#!/bin/env python

import tensorflow as tf
import numpy as np

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

x = tf.placeholder(tf.float32, shape=(1000, 30, 61))
y = tf.transpose(x, perm=[1, 0, 2])
z = tf.unstack(y, axis=0)
with tf.Session() as sess:
	data = np.random.rand(1000, 30, 61)
	ret = sess.run(x, feed_dict={x: data})
	print type(ret)
	print ret
	print "-------------------------"
	print sess.run(y, feed_dict={x: data})
	print "-------------------------"
	print sess.run(z, feed_dict={x: data})
	print "shape of z[0]"
	print sess.run(z[0], feed_dict={x: data})

