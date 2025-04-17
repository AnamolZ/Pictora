# freemiumApp.py

import os
import sys
import logging
import tensorflow as tf
from Processing import Processing
from Captioner import Captioner

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')
tf.get_logger().setLevel(logging.ERROR)

_model_cache = None

class App:
    @staticmethod
    def load_model():
        global _model_cache
        if _model_cache is None:
            print("Model not in RAM. Loading now...")
            _model_cache = tf.keras.models.load_model(
                '../models/caption',
                custom_objects={
                    'standardize': Processing.standardize,
                    'Captioner': Captioner
                },
                compile=False
            )
            if not hasattr(_model_cache, 'simple_gen'):
                _model_cache.simple_gen = Captioner.simple_gen.__get__(_model_cache, _model_cache.__class__)
            print("Custom Model loaded and cached in RAM.")
        else:
            print("Custom Model already loaded in RAM. Reusing it.")
        return _model_cache

    @staticmethod
    def run(image_path=None):
        model = App.load_model()
        caption = Processing.generate_caption(model, image_path)
        return caption
    
if __name__ == "__main__":
    image_path = os.path.abspath(os.path.join("../testImages", "image4.jpg"))
    print(App.run(image_path))