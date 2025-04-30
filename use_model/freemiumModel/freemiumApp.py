import os
import sys
import logging
import tensorflow as tf
from Processing import Processing
from Captioner import Captioner

from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')
tf.get_logger().setLevel(logging.ERROR)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_auth_token=HUGGINGFACE_TOKEN)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_auth_token=HUGGINGFACE_TOKEN).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model.to(device)

_model_cache = None

class App:
    @staticmethod
    def load_model():
        global _model_cache
        if _model_cache is None:
            print("Model not in RAM. Loading now...")
            _model_cache = tf.keras.models.load_model(
                os.path.join("..", "models", "caption"),
                custom_objects={"standardize": Processing.standardize, "Captioner": Captioner},
                compile=False
            )
            if not hasattr(_model_cache, "simple_gen"):
                _model_cache.simple_gen = Captioner.simple_gen.__get__(
                    _model_cache, _model_cache.__class__
                )
        else:
            pass
        return _model_cache

    @staticmethod
    def run(image_path):
        model = App.load_model()
        return Processing.generate_caption(model, image_path)

    @staticmethod
    def compute_blip_score(image, caption):
        inputs = processor(image, caption, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = blip_model(**inputs)
            logits = outputs.logits[0]
            return logits.softmax(dim=-1).max().item()

    @staticmethod
    def run_freemium_model(image_path, score_threshold=0.9, required_count=3):
        image = Image.open(image_path).convert("RGB")
        high_score_captions = []

        while len(high_score_captions) < required_count:
            try:
                caption = App.run(image_path)
                if not caption or not isinstance(caption, str):
                    continue
                score = App.compute_blip_score(image, caption)
                if score >= score_threshold:
                    high_score_captions.append((caption, score))
            except Exception as e:
                print(f"Error: {e}")

        best_caption, _ = min(high_score_captions, key=lambda x: len(x[0]))
        return best_caption

if __name__ == "__main__":
    path = os.path.abspath(os.path.join("..", "testImages", "image4.jpg"))
    print(App.run_freemium_model(path))
