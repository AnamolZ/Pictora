# model.py

import os
import sys

from ..freemiumModel.freemiumApp import App

# image_path = "../testimages/image4.jpg"

# assert os.path.exists(image_path), f"Image not found at: {image_path}"

def callfreemiumModel(image_path):
    return App.run_freemium_model(image_path)

def loadfreemiumModel():
    return App.load_model()

# if __name__ == "__main__":
#     callfreemiumModel(image_path)
