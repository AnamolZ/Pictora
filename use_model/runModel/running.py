from io import BytesIO
import os
import time
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File
from transformers import BlipProcessor, BlipForConditionalGeneration
from .loadModel import start_freemium_model_thread, get_freemium_model, load_blip_model
import warnings
from PIL import Image
from ..freemiumModel.freemiumApp import App
from fastapi import HTTPException
from PIL import Image
from io import BytesIO

warnings.filterwarnings("ignore")

app = FastAPI()

freemium_model = None
blip_model = None
blip_processor = None

@app.on_event("startup")
def startup_event():
    global freemium_model, blip_model, blip_processor
    print("Starting model loading...")

    start_time = time.time()
    start_freemium_model_thread()
    blip_model, blip_processor = load_blip_model()
    freemium_model = get_freemium_model(timeout=30)

    total_time = time.time() - start_time
    print(f"All models loaded in {total_time:.2f} seconds.")

@app.get("/model-status")
def model_status():
    return {
        "freemium_model": freemium_model is not None,
        "blip_model": blip_model is not None,
        "blip_processor": blip_processor is not None
    }