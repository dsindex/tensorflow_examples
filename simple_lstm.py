#!/bin/env python

import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def build_dictionary(sentences) :
    idx2char = []
    for sentence in sentences :
        for c in sentence :
            if c not in idx2char : idx2char.append(c)
    char2idx = {w: i for i, w in enumerate(idx2char)} # char to id
    return idx2char, char2idx

def one_hot(i, vocab_size) :
    return [ 1 if j == i else 0 for j in xrange(vocab_size) ]

def next_batch(sentences, begin, batch_size, sequence_length, char2idx) :
    batch_xs = []
    batch_ys = []
    count = 0
    for sentence in sentences[begin:] :
        x_data = sentence[0:sequence_length]
        vocab_size = len(char2idx)
        x_data = [char2idx[c] for c in x_data]
        x_data = [one_hot(i, vocab_size) for i in x_data]
        batch_xs.append(x_data)
        y_data = [char2idx[c] for c in sentence[1:sequence_length+1]]
        batch_ys.append(y_data)
        count += 1
        if count == batch_size : break
    batch_xs = np.array(batch_xs, dtype='f')      # [None, sequence_length, input_dim]
    batch_ys = np.array(batch_ys, dtype='int32')  # [None, sequence_length]
    return batch_xs, batch_ys, begin+count

def rnn_model(hidden_sizie, batch_size, X) :
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs, _states = tf.nn.dynamic_rnn(cell=cell, inputs=X, initial_state=initial_state, dtype=tf.float32)
    print outputs
    return outputs


sentences = ['abcdefg',
        'hijklmn',
        'opqrstu',
        'vwxyz**',
        'abcdefg',  # test
        'opqrstu']  # test
batch_size = 4

'''
sentences = ['hello world']
batch_size = 1
'''

# build dictionary
idx2char, char2idx = build_dictionary(sentences)
vocab_size = len(char2idx)

# config
learning_rate = 0.01
training_iters = 1000
input_dim = vocab_size                  # input dimension, one-hot size, vocab size
hidden_size = vocab_size                # output form LSTM, directly predict one-hot
sequence_length = len(sentences[0]) - 1 # time stpes

X = tf.placeholder(tf.float32, [None, sequence_length, hidden_size])  # X one-hot
Y = tf.placeholder(tf.int32, [None, sequence_length])                 # Y label

outputs = rnn_model(hidden_size, batch_size, X)
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    # training
    sess.run(tf.global_variables_initializer())
    begin = 0
    batch_xs, batch_ys, begin = next_batch(sentences, begin, batch_size, sequence_length, char2idx)
    step = 1
    while step < training_iters :
        l, _ = sess.run([loss, train], feed_dict={X: batch_xs, Y: batch_ys})
        results = sess.run(prediction, feed_dict={X: batch_xs})
        if step % 50 == 0 : 
            print(step, "loss:", l, "prediction: ", results, "true Y: ", batch_ys)
            for result in results :
                result_str = [idx2char[c] for c in np.squeeze(result)]
                print("\tPrediction str: ", ''.join(result_str))
        step += 1

    # inference
    print "-----------------------------------------"
    begin = 0
    batch_size = 4
    batch_xs, batch_ys, begin = next_batch(sentences, begin, batch_size, sequence_length, char2idx)
    feed_dict = {X: batch_xs, Y:batch_ys}
    results = sess.run(prediction, feed_dict={X: batch_xs})
    for result in results :
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))




