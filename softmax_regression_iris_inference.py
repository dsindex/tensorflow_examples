#!./bin/env python

from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np

# prediction function
X = tf.placeholder("float", [None, 5]) # row : infinity, col : 5 for x
W = tf.Variable(tf.zeros([5,3])) # row : 5 dimensions for x, col : 3 dimensions for y
y = tf.nn.softmax(tf.matmul(X, W)) # softmax, (None x 5) * ( 5 x 3 )

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver() # save all variables
checkpoint_dir = './'
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path :
    saver.restore(sess, ckpt.model_checkpoint_path)
else :
    sys.stderr.write("no checkpoint found" + '\n')
    sys.exit(-1)

p = sess.run(y, feed_dict={X:[[1,2,14,33,50]]}) # 1 0 0 -> type 0
print(p, sess.run(tf.arg_max(p, 1)))

p = sess.run(y, feed_dict={X:[[1,24,56,31,67]]}) # 0 1 0 -> type 1
print(p, sess.run(tf.arg_max(p, 1)))

p = sess.run(y, feed_dict={X:[[1,2,14,33,50], [1,24,56,31,67]]})
print(p, sess.run(tf.arg_max(p, 1)))

sess.close()
