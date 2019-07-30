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

# Some texts of different lengths.
english_sentences = ["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."]
japanese_sentences = ["犬", "子犬はいいです", "私は犬と一緒にビーチを散歩するのが好きです"]

# Graph set up.
g = tf.Graph()
with g.as_default():
  text_input = tf.placeholder(dtype=tf.string, shape=[None])
  embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-multilingual/1")
  embedded_text = embed(text_input)
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

# Initialize session.
session = tf.Session(graph=g)
session.run(init_op)

start_time = time.time()

# Compute embeddings.
en_result = session.run(embedded_text, feed_dict={text_input: english_sentences})
ja_result = session.run(embedded_text, feed_dict={text_input: japanese_sentences})

duration_time = time.time() - start_time
p = 'duration_time: ' + str(duration_time) + ' sec'
sys.stderr.write(p + '\n')

# Compute similarity matrix. Higher score indicates greater similarity.
similarity_matrix_ja = np.inner(en_result, ja_result)
print(similarity_matrix_ja)
'''
[[0.95641255 0.57370234 0.27371472]  -> dog : 犬
 [0.3986755  0.6300337  0.30383107]  -> Puppies are nice. : 子犬はいいです
 [0.25403124 0.21047902 0.8134757 ]] -> I enjoy taking long walks along the beach with my dog. : 私は犬と一緒にビーチを散歩するのが好きです
'''

# Some texts of different lengths.
q_sentences = ["오늘 날씨 어때"]
faq_sentences = ["오늘 판교 날씨 알려줘", "오늘 습도 알려줘", "판교까지 얼마나 걸려", "오늘 더워?"]

# Compute embeddings.
q_result = session.run(embedded_text, feed_dict={text_input: q_sentences})
faq_result = session.run(embedded_text, feed_dict={text_input: faq_sentences})

# Compute similarity matrix. Higher score indicates greater similarity.
similarity_matrix = np.inner(q_result, faq_result)
print(similarity_matrix)
'''
[[0.82803273 0.66021335 0.1759353  0.600948  ]] -> '오늘 날씨 어때' : '오늘 판교 날씨 알려줘'
'''
