import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from transformers import BlipProcessor, BlipForConditionalGeneration
from pickle import load
from .extract_features import FeatureExtractor
from .model import CaptioningModel
from .generate_caption import CaptionGenerator
from .clip_score import ClipScoreCalculator
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Trainer:
    def __init__(self, img_path, tokenizer_path, model_weights_path, max_length=32):
        self.img_path = img_path
        self.tokenizer = load(open(tokenizer_path, "rb"))
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.max_length = max_length
        self.feature_extractor = FeatureExtractor()
        self.model = CaptioningModel(self.vocab_size, self.max_length)
        self.model.model.load_weights(model_weights_path)
        self.caption_generator = CaptionGenerator(self.tokenizer, self.model.model, self.max_length)
        self.clip_score_calculator = ClipScoreCalculator()

    @staticmethod
    def create_sequences(tokenizer, max_length, desc, photo, vocab_size):
        X1, X2, y = [], [], []
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)

    def train(self):
        photo = self.feature_extractor.extract_features(self.img_path)
        photo = np.squeeze(photo)

        custom_caption = self.caption_generator.generate_desc(photo)

        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        img = Image.open(self.img_path)

        blip_inputs = blip_processor(images=img, return_tensors="pt")
        blip_output = blip_model.generate(**blip_inputs)
        blip_caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)

        custom_score = self.clip_score_calculator.compute_clip_score(img, custom_caption)
        blip_score = self.clip_score_calculator.compute_clip_score(img, blip_caption)

        caption_to_use = custom_caption if custom_score >= 22 else blip_caption

        corrected_caption = "start " + caption_to_use + " end"
        X1, X2, y = Trainer.create_sequences(self.tokenizer, self.max_length, corrected_caption, photo, self.vocab_size)

        for layer in self.model.model.layers[:-3]:
            layer.trainable = False
        self.model.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-5))
        self.model.train(X1, X2, y, epochs=5, batch_size=1)
        self.model.model.save_weights("models/modelV1.h5")

        print(f"Caption: {caption_to_use}")
