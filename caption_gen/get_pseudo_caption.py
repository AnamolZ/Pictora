import os
import csv
import warnings
from pathlib import Path
from typing import Dict

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, CLIPProcessor, CLIPModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from dotenv import load_dotenv

# Load .env variables
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN not found. Please check your .env file.")

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_AVAILABLE = torch.cuda.is_available()

class ImageCaptioningSystem:
    COCO_CLASSES = [
        "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
        "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ]
    TRANSFORM = T.Compose([T.ToTensor()])

    def __init__(self, images_path: Path, model_path: Path, output_csv: Path):
        self.images_path = images_path
        self.model_path = model_path
        self.output_csv = output_csv
        self.detection_model = self._load_detection_model()
        self.caption_generator = self._load_caption_generator()
        self.clip_processor, self.clip_model = self._load_clip_model()

    def _load_detection_model(self):
        model = fasterrcnn_resnet50_fpn(pretrained=False)
        return model.to(DEVICE).eval()

    def _load_caption_generator(self):
        if not self.model_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(
                "google/flan-t5-large",
                use_auth_token=HUGGINGFACE_TOKEN
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-large",
                use_auth_token=HUGGINGFACE_TOKEN
            )
            tokenizer.save_pretrained(self.model_path)
            model.save_pretrained(self.model_path)
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path).to(DEVICE)
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if GPU_AVAILABLE else -1)

    def _load_clip_model(self):
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(DEVICE).eval()
        return processor, model

    def _has_repeated_words(self, text: str) -> bool:
        words = text.split()
        return any(words.count(word) > 1 for word in words)

    def _detect_and_generate_captions(self) -> Dict[str, str]:
        captions = {}
        image_paths = list(self.images_path.glob("*.jpg"))
        for path in tqdm(image_paths, desc="Detect & Caption"):
            image = Image.open(path).convert("RGB")
            tensor = self.TRANSFORM(image).to(DEVICE)
            with torch.no_grad():
                outputs = self.detection_model([tensor])[0]
            labels = outputs["labels"].cpu().numpy()
            scores = outputs["scores"].cpu().numpy()
            detected_objects = [self.COCO_CLASSES[label] for label, score in zip(labels, scores) if score >= 0.6 and 0 <= label < len(self.COCO_CLASSES)]
            if detected_objects:
                caption_prompt = "Imagine an image using: " + ", ".join(detected_objects)
                caption = self.caption_generator(caption_prompt, max_length=40)[0]["generated_text"]
            else:
                caption = "No objects detected."
            if not self._has_repeated_words(caption):
                captions[path.name] = caption
        return captions

    def _compute_clip_similarity(self, image_path: Path, text: str) -> float:
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(text=[text], images=[image], return_tensors="pt", padding=True)
        for key, value in inputs.items():
            inputs[key] = value.to(DEVICE)
        outputs = self.clip_model(**inputs)
        image_embedding = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embedding = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return cosine_similarity(image_embedding.cpu().detach().numpy(), text_embedding.cpu().detach().numpy())[0][0]

    def _filter_by_similarity(self, captions: Dict[str, str], threshold: float = 0.28) -> Dict[str, str]:
        filtered_captions = {}
        for name, caption in tqdm(captions.items(), desc="Similarity Filter"):
            similarity_score = self._compute_clip_similarity(self.images_path / name, caption)
            if similarity_score > threshold:
                filtered_captions[name] = caption
        return filtered_captions

    def _remove_duplicate_captions(self, captions: Dict[str, str], threshold: float = 0.9) -> Dict[str, str]:
        items = list(captions.items())
        unique_captions = dict(items)
        for i, (name1, caption1) in enumerate(tqdm(items, desc="Duplicate Removal")):
            for name2, caption2 in items[i + 1:]:
                if name1 in unique_captions and name2 in unique_captions and caption1 == caption2:
                    score1 = self._compute_clip_similarity(self.images_path / name1, caption1)
                    score2 = self._compute_clip_similarity(self.images_path / name2, caption2)
                    unique_captions.pop(name1 if score1 < score2 else name2, None)
        return unique_captions

    def _clean_image_folder(self, captions: Dict[str, str]) -> Dict[str, str]:
        existing_files = set(os.listdir(self.images_path))
        valid_files = {name: caption for name, caption in captions.items() if name in existing_files}
        for filename in existing_files - set(valid_files):
            os.remove(self.images_path / filename)
        return valid_files

    def _save_captions_to_csv(self, captions: Dict[str, str]):
        with open(self.output_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image", "caption"])
            for name, caption in captions.items():
                writer.writerow([name, f" {caption}"])

    def run(self):
        captions = self._detect_and_generate_captions()
        captions = self._filter_by_similarity(captions)
        captions = self._remove_duplicate_captions(captions)
        captions = self._clean_image_folder(captions)
        self._save_captions_to_csv(captions)


if __name__ == "__main__":
    image_captioning_system = ImageCaptioningSystem(
        images_path=Path("../train_model/training_data/dataset"),
        model_path=Path("../use_model/models/flan-t5-large"),
        output_csv=Path("../train_model/training_data/pseudo_caption/pseudo_caption.csv"),
    )
    image_captioning_system.run()