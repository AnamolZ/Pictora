# clip_score.py

from transformers import CLIPProcessor, CLIPModel
import torch
import os
from dotenv import load_dotenv
load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

class ClipScoreCalculator:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_auth_token=HUGGINGFACE_TOKEN)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_auth_token=HUGGINGFACE_TOKEN)

    def compute_clip_score(self, image, text):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(device)
        inputs = self.clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item()