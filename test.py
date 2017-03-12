#!/bin/env python

import tensorflow as tf

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
	# Run the 'init' op
	sess.run(init_op)
	# Print the initial value of 'state'
	print sess.run(state)
	# Run the op that updates 'state' and print 'state'.
	for _ in range(3):
		sess.run(update)
		print sess.run(state)

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)
with tf.Session() as sess:
	result = sess.run([mul, intermed])
	print result

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
	print sess.run([output], feed_dict={input1:[7.], input2:[2.]})
