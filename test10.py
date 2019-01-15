#!/bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np

# data is [[0, 1, 2, 3, 4, 5],
#          [6, 7, 8, 9, 10, 11],
#          [12 13 14 15 16 17],
#          [18 19 20 21 22 23],
#          [24, 25, 26, 27, 28, 29]]

data = np.reshape(np.arange(30), [5, 6])
x = tf.constant(data)

# [ 8 27 17]
result1 = tf.gather_nd(x, [[1, 2], [4, 3], [2, 5]])

# treat [0, 0] as padding, and ignore all [i, 0]
# [[1 2 4 0]
#  [7 9 0 0]]
result2 = tf.gather_nd(x, [[[0, 1], [0, 2], [0, 4], [0, 0]],
                           [[1, 1], [1, 3], [0, 0], [0, 0]],])

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    out = sess.run(result2)
    print(out)
