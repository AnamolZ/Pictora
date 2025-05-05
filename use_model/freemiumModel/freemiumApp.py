import os
import sys
import logging
import tensorflow as tf
from .Processing import Processing
from .Captioner import Captioner

from PIL import Image
import torch
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')
tf.get_logger().setLevel(logging.ERROR)

model_dir = os.path.abspath(os.path.join("..", "models", "blip"))
device = torch.device("cpu")

def ensure_blip_model():
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    files_required = [
        os.path.join(model_dir, "config.json"),
        os.path.join(model_dir, "pytorch_model.bin"),
        os.path.join(model_dir, "preprocessor_config.json")
    ]
    
    missing_files = [f for f in files_required if not os.path.exists(f)]

    if missing_files:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_auth_token=HUGGINGFACE_TOKEN)
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", use_auth_token=HUGGINGFACE_TOKEN)

        processor.save_pretrained(model_dir)
        model.save_pretrained(model_dir)

ensure_blip_model()

try:
    processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
except Exception as e:
    print(f"Error loading processor: {e}")

try:
    blip_model = BlipForConditionalGeneration.from_pretrained(model_dir, local_files_only=True).eval()
except Exception as e:
    print(f"Error loading model: {e}")

blip_model.to(device)

_model_cache = None

freemium_modelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "caption")
freemium_modelPath = os.path.normpath(freemium_modelPath)

class App:
    @staticmethod
    def load_model():
        global _model_cache
        if _model_cache is None:
            print("Model not in RAM. Loading now...")
            _model_cache = tf.keras.models.load_model(
                freemium_modelPath,
                custom_objects={"standardize": Processing.standardize, "Captioner": Captioner},
                compile=False
            )
            if not hasattr(_model_cache, "simple_gen"):
                _model_cache.simple_gen = Captioner.simple_gen.__get__(
                    _model_cache, _model_cache.__class__
                )
        return _model_cache

    @staticmethod
    def run():
        return App.load_model()

    @staticmethod
    def compute_blip_score(image, caption):
        inputs = processor(image, caption, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = blip_model(**inputs)
            logits = outputs.logits[0]
            return logits.softmax(dim=-1).max().item()

    @staticmethod
    def run_freemium_model(image_path, score_threshold=0.9, required_count=5, total_captions=20):
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        model = App.run()
        high_score_captions = []

        def generate_and_score():
            try:
                caption = Processing.generate_caption(model, image_path)
                if not caption or not isinstance(caption, str):
                    return None
                score = App.compute_blip_score(image, caption)
                if score >= score_threshold:
                    return (caption, score)
            except:
                return None
            return None

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(generate_and_score) for _ in range(total_captions)]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    high_score_captions.append(result)
                    if len(high_score_captions) >= required_count:
                        break

        if not high_score_captions:
            return None

        # Sort captions by score DESC, then length ASC
        sorted_captions = sorted(high_score_captions, key=lambda x: (-x[1], len(x[0])))
        best_caption = sorted_captions[0][0]

        duration = time.time() - start_time
        print(f"Best caption generated in {duration:.2f} seconds with score {sorted_captions[0][1]:.4f} and length {len(best_caption)}.")
        return best_caption

if __name__ == "__main__":
    path = os.path.abspath(os.path.join("..", "testImages", "image4.jpg"))
    print(App.run_freemium_model(path))
