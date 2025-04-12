import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from keras.applications.inception_v3 import preprocess_input

class FeatureExtractor:
    def extract_features(self, directory, model):
        features = {}
        valid_images = ['.jpg', '.jpeg', '.png']
        valid_files = [img for img in os.listdir(directory)
                       if os.path.splitext(img)[1].lower() in valid_images]
        with tqdm(total=len(valid_files), desc="Extracting features", position=0, leave=True) as pbar:
            for img in valid_files:
                filename = os.path.join(directory, img)
                image = Image.open(filename).resize((299, 299))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = np.array(image)
                if image.shape[-1] == 4:
                    image = image[..., :3]
                image = np.expand_dims(image, axis=0)
                image = preprocess_input(image)
                feature = model.predict(image, verbose=0)
                features[img] = feature
                pbar.update(1)
        return features

    def load_features(self, photos, features_file):
        from pickle import load
        all_features = load(open(features_file, "rb"))
        return {k: all_features[k] for k in photos if k in all_features}
