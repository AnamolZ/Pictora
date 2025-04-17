#blip_trainer.py

import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from clip_score import ClipScoreCalculator
import sys

sys.path.append(os.path.abspath('../runModel'))

from freemium_infer import callfreemiumModel

class Trainer:
    blip_model = None
    blip_processor = None

    def __init__(self, img_path):
        self.img_path = img_path
        self.clip_score_calculator = ClipScoreCalculator()
        self.model_dir = os.path.abspath("../models/blip")
        self._load_blip_to_ram()

    def _load_blip_to_ram(self):
        if Trainer.blip_model and Trainer.blip_processor:
            return

        if os.path.exists(self.model_dir):
            Trainer.blip_processor = BlipProcessor.from_pretrained(self.model_dir)
            Trainer.blip_model = BlipForConditionalGeneration.from_pretrained(self.model_dir)
        else:
            Trainer.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            Trainer.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            os.makedirs(self.model_dir, exist_ok=True)
            Trainer.blip_processor.save_pretrained(self.model_dir)
            Trainer.blip_model.save_pretrained(self.model_dir)

    def train(self):
        img = Image.open(self.img_path)
        blip_inputs = Trainer.blip_processor(images=img, return_tensors="pt")
        blip_output = Trainer.blip_model.generate(**blip_inputs)
        blip_caption = Trainer.blip_processor.decode(blip_output[0], skip_special_tokens=True)

        custom_caption = callfreemiumModel(self.img_path)
        custom_caption = str(custom_caption)
        custom_score = self.clip_score_calculator.compute_clip_score(img, custom_caption)
        return custom_caption if custom_score >= 22 else blip_caption