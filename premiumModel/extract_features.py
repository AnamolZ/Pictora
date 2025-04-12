from keras.applications.xception import Xception, preprocess_input
import numpy as np
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        self.model = Xception(include_top=False, pooling="avg")

    def extract_features(self, filename):
        img = Image.open(filename).resize((299, 299))
        img = np.array(img)
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feature = self.model.predict(img)
        return feature
