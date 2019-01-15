#!/bin/env python

'''
reference : https://www.tensorflow.org/versions/r0.8/how_tos/distributed/index.html
'''

from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("model_path", "/root/tensorflow/train_logs", "path to save model")
FLAGS = tf.app.flags.FLAGS

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    print(cluster._cluster_spec)

    # Create and start a server for the local task.
    #server = tf.train.Server(cluster.as_cluster_def(), job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps" :
        server.join()
    elif FLAGS.job_name == "worker":
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=cluster)):
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

            loss = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y)))
            global_step = tf.Variable(0)
            train_op = tf.train.AdagradOptimizer(0.01).minimize(loss, global_step=global_step)
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        # Create a "supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=FLAGS.model_path,
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        sess = sv.prepare_or_wait_for_session(server.target)

        # Start queue runners for the input pipelines (if any).
        sv.start_queue_runners(sess)

        # Loop until the supervisor shuts down (or 1000000 steps have completed).
        mnist = input_data.read_data_sets("./MNIST-data/", one_hot=True)
        step = 0
        while not sv.should_stop() and step < 10000:
            # Run a training step asynchronously.
            # See `tf.train.SyncReplicasOptimizer` for additional details on how to
            # perform *synchronous* training.
            batch_xs, batch_ys = mnist.train.next_batch(50)
            if step % 100 == 0:
                print("job : %s/%s" % (FLAGS.job_name,FLAGS.task_index), "step : ", step, ",training accuracy :", sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
            _, step = sess.run([train_op, global_step], feed_dict={x: batch_xs, y_: batch_ys})

        # save model
        saver.save(sess, FLAGS.model_path + "/mlp.ckpt")

        # Ask for all the services to stop.
        sv.stop()

if __name__ == "__main__":
  tf.app.run()

