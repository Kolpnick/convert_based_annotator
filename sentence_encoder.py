import tensorflow_hub as tfhub
import tensorflow_text
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os


tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
CONVERT_DIR = os.environ.get("CONVERT_DIR", None)


class SentenceEncoder:
    def __init__(self,
                 multiple_contexts=True,
                 batch_size=32,
                 ):

        self.multiple_contexts = multiple_contexts
        self.batch_size = batch_size

        self.sess = tf.compat.v1.Session()
        self.text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=[None])

        self.module = tfhub.Module(CONVERT_DIR)
        self.context_encoding_tensor = self.module(self.text_placeholder, signature="encode_context")
        self.encoding_tensor = self.module(self.text_placeholder)

        self.response_encoding_tensor = self.module(self.text_placeholder, signature="encode_response")
        self.sess.run(tf.compat.v1.tables_initializer())
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def encode_sentences(self, sentences):
        return self.batch_process(lambda x: self.sess.run(
            self.encoding_tensor, feed_dict={self.text_placeholder: x}
        ), sentences)

    def encode_contexts(self, sentences):
        return self.batch_process(lambda x: self.sess.run(
            self.context_encoding_tensor, feed_dict={self.text_placeholder: x}
              ), sentences)

    def encode_responses(self, sentences):
        return self.batch_process(
            lambda x: self.sess.run(
                self.response_encoding_tensor, feed_dict={self.text_placeholder: x}
            ),
            sentences)

    def batch_process(self, func, sentences):
        encodings = []
        for i in tqdm(range(0, len(sentences), self.batch_size), disable=True):
            encodings.append(func(sentences[i:i + self.batch_size]))
        return SentenceEncoder.l2_normalize(np.vstack(encodings))

    @staticmethod
    def l2_normalize(encodings):
        norms = np.linalg.norm(encodings, ord=2, axis=-1, keepdims=True)
        return encodings / norms
