#!/bin/env python

import tensorflow as tf
import numpy as np

a = tf.constant([[1, 2], [3, 4]])

# Reshape `a` as a vector. -1 means "set this dimension automatically".
a_as_vector = tf.reshape(a, [-1])

# Create another vector containing zeroes to pad `a` to (2 * 3) elements.
zero_padding = tf.zeros([2 * 3] - tf.shape(a_as_vector), dtype=a.dtype)

# Concatenate `a_as_vector` with the padding.
a_padded = tf.concat([a_as_vector, zero_padding], 0)

# Reshape the padded vector to the desired shape.
result = tf.reshape(a_padded, [2, 3])

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(a)
    print sess.run(result)
