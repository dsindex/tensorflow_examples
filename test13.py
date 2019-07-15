#!/bin/env python

from __future__ import print_function
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece

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

# Compute embeddings.
en_result = session.run(embedded_text, feed_dict={text_input: english_sentences})
ja_result = session.run(embedded_text, feed_dict={text_input: japanese_sentences})

# Compute similarity matrix. Higher score indicates greater similarity.
similarity_matrix_ja = np.inner(en_result, ja_result)
print(similarity_matrix_ja)
