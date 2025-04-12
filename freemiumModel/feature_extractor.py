from keras.applications.xception import Xception, preprocess_input
from PIL import Image
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.model = Xception(include_top=False, pooling="avg")

    def extract(self, image_path):
        img = Image.open(image_path).resize((299, 299))
        img = np.array(img)
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return self.model.predict(img)
