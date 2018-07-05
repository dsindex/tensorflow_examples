#!/bin/env python

import tensorflow as tf

x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])

s0 = tf.stack([x,y,z], axis=0)   # pack along first dim
s1 = tf.stack([x,y,z], axis=1)   # pack along second dim

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print [x,y,z]
	print sess.run(s0)
	print sess.run(s1)
	print sess.run(tf.sign([-1, -0.2, 0.3]))

