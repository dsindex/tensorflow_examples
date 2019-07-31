#!/bin/env python

from __future__ import print_function
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

import sys
import time

'''
python -m pip install tensorflow-gpu==1.11.0
python -m pip install tensorflow-hub==0.4.0
python -m pip install tf_sentencepiece==0.1.6
'''

# Create graph and session
g = tf.Graph()
with g.as_default():
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()
session = tf.Session(graph=g)
session.run(init_op)

# Warmup 
warmup_sentences = ["this is an warmup sentence.", "one more?"]
start_time = time.time()
session.run(embedded_text, feed_dict={text_input: warmup_sentences})
duration_time = time.time() - start_time
p = 'duration_time(warmup data embeddings): ' + str(duration_time) + ' sec'
sys.stderr.write(p + '\n')
                     
# Prepare target data
faq_sentences = ["권진아의 fly away 틀어",
                 "인기 많은 일렉 음악 틀어줘",
                 "랜덤으로 들려줘",
                 "캐롤 동요 들려줘",
                 "아이유 좋은날 틀어줘",
                 "신나는 댄스 음악",
                 "이 노래로 알람 음악 설정",
                 "영어 공부에 도움되는 거 들려줘",
                 "처음부터 다시 들려줘",
                 "지금 나오는 노래 제목이 뭐야",
                 "불꽃 심장 노래 틀어줘"]

# Precomputing embeddings
start_time = time.time()
faq_result = session.run(embedded_text, feed_dict={text_input: faq_sentences})
duration_time = time.time() - start_time
p = 'duration_time(target data embeddings): ' + str(duration_time) + ' sec'
sys.stderr.write(p + '\n')

# Prepare query embedding
q_sentences = ["강명식 노래 틀어줘"]
start_time = time.time()
q_result = session.run(embedded_text, feed_dict={text_input: q_sentences})
duration_time = time.time() - start_time
p = 'duration_time(query embedding): ' + str(duration_time) + ' sec'
sys.stderr.write(p + '\n')

# Compute similarity matrix. Higher score indicates greater similarity.
start_time = time.time()
similarity_matrix = np.inner(q_result, faq_result)
print(similarity_matrix)
duration_time = time.time() - start_time
p = 'duration_time(similarity): ' + str(duration_time) + ' sec'
sys.stderr.write(p + '\n')

"""
GPU : TITAN X Pascal
CPU : cores 8, hyper 4, processors 32, Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz
MEM : 128G

GPU

duration_time(warmup data embeddings): 1.4113376140594482 sec
duration_time(target data embeddings): 0.007622241973876953 sec
duration_time(query embedding): 0.00807642936706543 sec
[[0.13228425 0.38765287 0.36059707 0.28463903 0.19212598 0.43535215
  0.44282746 0.09915389 0.2527572  0.43614778 0.47565958]]
duration_time(similarity): 0.0012204647064208984 sec

CPU

duration_time(warmup data embeddings): 1.0827670097351074 sec
duration_time(target data embeddings): 0.011441469192504883 sec
duration_time(query embedding): 0.006826877593994141 sec
[[0.13228424 0.38765296 0.36059725 0.28463918 0.19212595 0.4353522
  0.44282752 0.09915397 0.2527572  0.4361478  0.4756597 ]]
duration_time(similarity): 0.0008025169372558594 sec

"""
