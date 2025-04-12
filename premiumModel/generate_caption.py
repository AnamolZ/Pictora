from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

class CaptionGenerator:
    def __init__(self, tokenizer, model, max_length):
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

    def word_for_id(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def generate_desc(self, photo):
        in_text = 'start'
        for _ in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            sequence = np.array(sequence)
            pred = self.model.predict([np.array([photo]), sequence], verbose=0)
            pred = np.argmax(pred)
            word = self.word_for_id(pred)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        return in_text
