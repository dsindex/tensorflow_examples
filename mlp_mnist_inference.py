#!./bin/env python

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

mnist = input_data.read_data_sets("./MNIST-data/", one_hot=True)

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

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver() # save all variables
checkpoint_dir = './train_logs/'
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path :
    print ckpt.model_checkpoint_path
    saver.restore(sess, ckpt.model_checkpoint_path)
else :
    sys.stderr.write("no checkpoint found" + '\n')
    sys.exit(-1)

test_xs, test_ys = mnist.test.next_batch(10000)
test_accuracy = sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys}) 
print "test accuracy : ", test_accuracy

sess.close()
