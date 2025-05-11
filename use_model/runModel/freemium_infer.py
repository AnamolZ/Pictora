# freemium_infer.py

from ..freemiumModel.freemiumApp import freemiumApp

def callfreemiumModel(image_path):
    return freemiumApp.generate_single_caption(image_path)