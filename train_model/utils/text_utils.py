import re
import string
import tensorflow as tf

class TextUtils:
    def __init__(self, vocabulary_size=5000):
        self.vocabulary_size = vocabulary_size
        self.tokenizer = None
        self.word_to_index = None
        self.index_to_word = None
    
    def standardize(self, s):
        s = tf.strings.lower(s)
        s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
        s = tf.strings.join(['[START]', s, '[END]'], separator=' ')
        return s
    
    def setup_tokenizer(self):
        self.tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocabulary_size,
            standardize=self.standardize,
            ragged=True)
    
    def adapt_tokenizer(self, dataset):
        self.tokenizer.adapt(dataset.map(lambda fp, txt: txt).unbatch().batch(1024))
        
        self.word_to_index = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=self.tokenizer.get_vocabulary())
        
        self.index_to_word = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=self.tokenizer.get_vocabulary(),
            invert=True)
