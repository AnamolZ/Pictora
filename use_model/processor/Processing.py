# Processing.py

import tensorflow as tf
import string
import re

class Processing:
    @staticmethod
    def standardize(s):
        s=tf.strings.lower(s)
        s=tf.strings.regex_replace(s,f'[{re.escape(string.punctuation)}]','')
        s=tf.strings.join(['[START]',s,'[END]'],separator=' ')
        return s

    @staticmethod
    def preprocess_image(path):
        img=tf.io.read_file(path)
        img=tf.image.decode_jpeg(img,channels=3)
        img=tf.image.resize(img,(224,224))
        return tf.keras.applications.mobilenet_v3.preprocess_input(img)

    @staticmethod
    def generate_caption(model,path,temperature=0.5):
        img=Processing.preprocess_image(path)
        img=tf.expand_dims(img,0)
        return model.simple_gen(img,temperature)