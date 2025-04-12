import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SequenceGenerator:
    def create_sequences(self, tokenizer, max_length, desc_list, feature, vocab_size):
        X1, X2, y = [], [], []
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(feature)
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)
