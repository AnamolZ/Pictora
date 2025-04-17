# model.py

import os
import sys

sys.path.append(os.path.abspath('../freemiumModel'))

from freemiumApp import App

# image_path = "../testimages/image4.jpg"

# assert os.path.exists(image_path), f"Image not found at: {image_path}"

def callfreemiumModel(image_path):
    return App.run(image_path)

# if __name__ == "__main__":
#     callfreemiumModel(image_path)
