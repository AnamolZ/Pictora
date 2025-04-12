from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class CaptionGenerator:
    def __init__(self, model, tokenizer, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _word_for_id(self, integer):
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def generate(self, photo):
        in_text = 'start'
        for _ in range(self.max_length):
            sequence = self.tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=self.max_length)
            yhat = self.model.predict([photo, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self._word_for_id(yhat)
            if word is None:
                break
            in_text += ' ' + word
            if word == 'end':
                break
        return in_text.replace("start", "").replace("end", "").strip()
