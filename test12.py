#!/bin/env python

from __future__ import print_function
import tensorflow as tf
import numpy as np

with tf.name_scope("my_scope1"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # var1:0
print(v2.name)  # my_scope1/var2:0
print(a.name)   # my_scope1/Add:0

with tf.variable_scope("my_scope2"):
    v1 = tf.get_variable("var1", [1], dtype=tf.float32)
    v2 = tf.Variable(1, name="var2", dtype=tf.float32)
    a = tf.add(v1, v2)

print(v1.name)  # my_scope2/var1:0
print(v2.name)  # my_scope2/var2:0
print(a.name)   # my_scope2/Add:0

with tf.name_scope("foo"):
    with tf.variable_scope("var_scope"):
        v = tf.get_variable("var", [1])
with tf.name_scope("bar"):
    with tf.variable_scope("var_scope", reuse=True):
        v1 = tf.get_variable("var", [1])
assert v1 == v
print(v.name)   # var_scope/var:0
print(v1.name)  # var_scope/var:0

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
