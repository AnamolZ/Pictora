import os
import sys
import logging
import threading
import tensorflow as tf
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from ..freemiumModel.Processing import Processing
from ..freemiumModel.Captioner import Captioner
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.stderr = open(os.devnull, 'w')
tf.get_logger().setLevel(logging.ERROR)

freemium_model_cache = None
freemium_model_loaded_event = threading.Event()

blip_model = None
blip_processor = None

base_dir = os.path.dirname(os.path.abspath(__file__))
freemium_model_path = os.path.normpath(os.path.join(base_dir, "..", "models", "caption"))
premium_model_path = os.path.normpath(os.path.join(base_dir, "..", "models", "blip"))

lock = threading.Lock()

def _freemium_model_thread():
    global freemium_model_cache
    with lock:
        try:
            print("Background loading freemium (custom) model into RAM...")
            model = tf.keras.models.load_model(
                freemium_model_path,
                custom_objects={"standardize": Processing.standardize, "Captioner": Captioner},
                compile=False
            )
            if not hasattr(model, "simple_gen"):
                model.simple_gen = Captioner.simple_gen.__get__(model, model.__class__)
            freemium_model_cache = model
            freemium_model_loaded_event.set()
        except Exception as e:
            print(f"Error loading freemium model: {e}")
            freemium_model_cache = None
            freemium_model_loaded_event.set()

def start_freemium_model_thread():
    thread = threading.Thread(target=_freemium_model_thread, daemon=True)
    thread.start()

def get_freemium_model():
    freemium_model_loaded_event.wait()
    if freemium_model_cache is None:
        raise RuntimeError("Freemium model not loaded.")
    return freemium_model_cache

def load_blip_model(model_dir=premium_model_path):
    global blip_model, blip_processor
    with lock:
        if blip_model is not None and blip_processor is not None:
            print("BLIP model already loaded.")
            return blip_model, blip_processor
        model_dir = os.path.abspath(model_dir)
        try:
            if os.path.exists(model_dir):
                blip_processor = BlipProcessor.from_pretrained(model_dir, use_fast=False)
                blip_model = BlipForConditionalGeneration.from_pretrained(model_dir)
            else:
                print("Downloading BLIP model from HuggingFace...")
                blip_processor = BlipProcessor.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    use_auth_token=HUGGINGFACE_TOKEN,
                    use_fast=False
                )
                blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    use_auth_token=HUGGINGFACE_TOKEN
                )
                os.makedirs(model_dir, exist_ok=True)
                blip_processor.save_pretrained(model_dir)
                blip_model.save_pretrained(model_dir)
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
        return blip_model, blip_processor

def load_models_concurrently():
    start_freemium_model_thread()
    thread_blip = threading.Thread(target=load_blip_model)
    thread_blip.start()
    thread_blip.join()
    freemium_model_loaded_event.wait()

if __name__ == "__main__":
    load_models_concurrently()
