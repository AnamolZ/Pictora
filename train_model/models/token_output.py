import collections
import numpy as np
import tensorflow as tf
import tqdm

class TokenOutput(tf.keras.layers.Layer):
    def __init__(self, tokenizer, banned_tokens=('', '[UNK]', '[START]'), **kwargs):
        super().__init__()
        self.dense = tf.keras.layers.Dense(units=tokenizer.vocabulary_size(), **kwargs)
        self.tokenizer = tokenizer
        self.banned_tokens = banned_tokens
        self.bias = None

    def adapt(self, dataset):
        counter = collections.Counter()
        vocab = {tok: idx for idx, tok in enumerate(self.tokenizer.get_vocabulary())}
        for token_batch in tqdm.tqdm(dataset):
            counter.update(token_batch.numpy().flatten())
        counts = np.zeros(self.tokenizer.vocabulary_size())
        for key, value in counter.items():
            counts[key] = value
        for banned in self.banned_tokens:
            counts[vocab[banned]] = 0
        total = counts.sum()
        prob = counts / total
        prob[counts == 0] = 1.0
        log_prob = np.log(prob)
        self.bias = log_prob
        self.bias[counts == 0] = -1e9

    def call(self, inputs):
        output = self.dense(inputs)
        return output + self.bias
