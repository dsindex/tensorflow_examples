#!/bin/env python

import tensorflow as tf

x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])

m = tf.Variable([x,y,z])
mt = tf.convert_to_tensor(m)
s0 = tf.stack(mt, axis=0)   # pack along first dim
#s1 = tf.stack(mt, axis=1)   # pack along second dim

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init_op)
	print x
	print m
	print sess.run(m)
	print mt
	print sess.run(mt)
	print sess.run(s0)
	print mt
	#print sess.run(s1)

