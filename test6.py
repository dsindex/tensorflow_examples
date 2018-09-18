from __future__ import print_function
import tensorflow as tf
import numpy as np

def normalize(inputs,
              epsilon = 1e-8,
              scope="ln",
              reuse=None):
    """Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        print(params_shape)
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        print(mean)
        print(variance)
        beta= tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        print(beta)
        print(gamma)
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta

    return outputs


# 1 x 5 x 4
inputs = tf.constant([ [ [1, 1, 1, 0],
                         [1, 0.5, 0, 0],
                         [0.3, 0.2, 0.5, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0 ] ] ], dtype=tf.float32)

outputs = normalize(inputs)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(outputs))
