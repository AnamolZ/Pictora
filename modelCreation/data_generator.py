import numpy as np
import tensorflow as tf
from .sequence_generator import SequenceGenerator

class DataGenerator:
    def data_generator(self, descriptions, features, tokenizer, max_length, vocab_size):
        sg = SequenceGenerator()
        def generator():
            keys = list(descriptions.keys())
            np.random.shuffle(keys)
            for key in keys:
                feature = features[key][0]
                X1, X2, y = sg.create_sequences(tokenizer, max_length, descriptions[key], feature, vocab_size)
                for i in range(len(X1)):
                    yield {'input_1': X1[i], 'input_2': X2[i]}, y[i]
        output_signature = (
            {
                'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
                'input_2': tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
            },
            tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
        )
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        return dataset.repeat().batch(32)

    def get_steps_per_epoch(self, descriptions):
        total = 0
        for key in descriptions:
            for desc in descriptions[key]:
                total += len(desc.split()) - 1
        return max(1, total // 32)
