# blip_score.py

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class BlipScoreCalculator:
    def __init__(self):
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_auth_token=HUGGINGFACE_TOKEN
        ).eval()
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_auth_token=HUGGINGFACE_TOKEN
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blip_model.to(self.device)

    def compute_blip_score(self, image, caption):
        inputs = self.blip_processor(image, caption, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.blip_model(**inputs)
            logits = outputs.logits[0]
            return logits.softmax(dim=-1).max().item()
