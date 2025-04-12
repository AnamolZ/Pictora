import os
from .description import DescriptionProcessor
from .feature_extractor import FeatureExtractor
from .tokenizer_manager import TokenizerManager
from .data_generator import DataGenerator
from .model_builder import ModelBuilder
from keras.applications.xception import Xception
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pickle import dump, load
import os
import numpy as np

class TrainingPipeline:
    def __init__(self):
        self.desc = DescriptionProcessor()
        self.fe = FeatureExtractor()
        self.tk = TokenizerManager()
        self.dg = DataGenerator()
        self.mb = ModelBuilder()
        self.dataset_text = "training_data/pseudo_caption"
        self.dataset_images = "training_data/dataset"
        self.token_file = "processed/tokenizer.p"
        self.desc_file = "processed/descriptions.txt"
        self.features_file = "processed/features.p"
        self.caption_file = os.path.join(self.dataset_text, "pseudo_caption.txt")

    def prepare(self):
        if not os.path.exists('processed'):
            os.makedirs('processed')

        for file in [self.desc_file, self.token_file, self.features_file]:
            if os.path.exists(file):
                os.remove(file)

        descriptions = self.desc.all_img_captions(self.caption_file)
        descriptions = self.desc.cleaning_text(descriptions)
        self.desc.save_descriptions(descriptions, self.desc_file)

        model_extractor = Xception(include_top=False, pooling='avg', weights="imagenet")
        features = self.fe.extract_features(self.dataset_images, model_extractor)
        dump(features, open(self.features_file, "wb"))

        photos = list(features.keys())
        descriptions = self.desc.load_clean_descriptions(self.desc_file, photos)
        tokenizer = self.tk.create_tokenizer(descriptions)
        dump(tokenizer, open(self.token_file, "wb"))

        return descriptions, features, tokenizer

    def train(self):
        descriptions, features, tokenizer = self.prepare()
        vocab_size = len(tokenizer.word_index) + 1

        all_desc = [d for descs in descriptions.values() for d in descs]
        max_length_val = max(len(d.split()) for d in all_desc)

        photos = list(features.keys())
        np.random.shuffle(photos)
        split_idx = int(0.8 * len(photos))
        train_ids = photos[:split_idx]
        val_ids = photos[split_idx:]

        train_descriptions = {k: descriptions[k] for k in train_ids}
        val_descriptions = {k: descriptions[k] for k in val_ids}
        train_features = {k: features[k] for k in train_ids}
        val_features = {k: features[k] for k in val_ids}

        train_dataset = self.dg.data_generator(train_descriptions, train_features, tokenizer, max_length_val, vocab_size)
        val_dataset = self.dg.data_generator(val_descriptions, val_features, tokenizer, max_length_val, vocab_size)
        steps_train = self.dg.get_steps_per_epoch(train_descriptions)
        steps_val = self.dg.get_steps_per_epoch(val_descriptions)

        model = self.mb.define_model(vocab_size, max_length_val)

        if not os.path.exists("models"):
            os.mkdir("models")

        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        mc = ModelCheckpoint('models/modelV1.h5', monitor='val_loss', save_best_only=True)

        model.fit(
            train_dataset,
            epochs=20,
            steps_per_epoch=steps_train,
            validation_data=val_dataset,
            validation_steps=steps_val,
            callbacks=[es, mc],
            verbose=1
        )
