#!/bin/env python

import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow_serving.example import mnist_input_data

tf.app.flags.DEFINE_string('input_path', './MNIST_data', 'input directory path')
tf.app.flags.DEFINE_integer('training_iteration', 10000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('export_path', './export', 'export directory path')
FLAGS = tf.app.flags.FLAGS

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

mnist = mnist_input_data.read_data_sets(FLAGS.input_path, one_hot=True)

x = tf.placeholder(tf.float32, [None, 28*28])
y_ = tf.placeholder(tf.float32, [None, 10])

# hidden layer
W_h1 = weight_variable([28*28, 512])
b_h1 = bias_variable([512])
h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_h1)

# output layer
W_out = weight_variable([512, 10])
b_out = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h1, W_out) + b_out)

# training
cost = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
NUM_THREADS = 5
sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS,inter_op_parallelism_threads=NUM_THREADS,log_device_placement=False))
init = tf.global_variables_initializer()
sess.run(init)
for i in range(FLAGS.training_iteration):
	batch_xs, batch_ys = mnist.train.next_batch(50)
	if i % 100 == 0:
		print "step : ", i, "training accuracy :", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys})	
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# export model
export_path = FLAGS.export_path
print 'Exporting trained model to', export_path
saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=x, scores_tensor=y)
model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)
print 'Done exporting!'

sess.close()
