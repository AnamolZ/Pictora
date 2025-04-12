from tensorflow.keras.preprocessing.text import Tokenizer

class TokenizerManager:
    def create_tokenizer(self, descriptions):
        all_desc = []
        for key in descriptions:
            all_desc.extend(descriptions[key])
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(all_desc)
        return tokenizer
