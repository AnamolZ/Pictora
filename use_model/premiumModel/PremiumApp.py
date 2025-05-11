# premiumApp.py

from PIL import Image
from ..qualityCheck.clip_score import ClipScoreCalculator
from ..freemiumModel.freemiumApp import freemiumApp

class premiumApp:
    model = None

    @staticmethod
    def inject_model(external_model):
        premiumApp.model = external_model

    def __init__(self, img_path):
        self.img_path = img_path
        self.clip_score_calculator = ClipScoreCalculator()

    def run_premium_model(self):
        if premiumApp.model is None:
            raise RuntimeError("Model has not been injected")

        custom_caption = str(freemiumApp.run_freemium_model(self.img_path))
        return custom_caption