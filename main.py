# Version 1, Apporach CLIP with Additional LLM, Phase 1

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import random
from PIL import Image
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, ColorJitter, RandomRotation
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np

with open("data/captions.json", "r") as f:
    captions_data = json.load(f)

def create_positive_negative_pairs(captions_data):
    positive_pairs, negative_pairs = [], []
    for entry in captions_data:
        image_id, file_name, captions = entry["image_id"], entry["file_name"], entry["captions"]
        positive_caption = random.choice(captions)
        positive_pairs.append((image_id, file_name, positive_caption, 1))
        random_entry = random.choice(captions_data)
        while random_entry["image_id"] == image_id:
            random_entry = random.choice(captions_data)
        negative_caption = random.choice(random_entry["captions"])
        negative_pairs.append((image_id, file_name, negative_caption, 0))
    return positive_pairs, negative_pairs

positive_pairs, negative_pairs = create_positive_negative_pairs(captions_data)
_pairs = positive_pairs + negative_pairs
random.shuffle(_pairs)
train_pairs, test_pairs = train_test_split(_pairs, test_size=0.2, random_state=42)
train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.25, random_state=42)

train_transform = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    RandomRotation(10),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
clip_transform = Compose([Resize((224, 224)), ToTensor()])

def load_image(file_name):
    try:
        image_path = f"data/images/{file_name}"
        image = Image.open(image_path).convert("RGB")
        raw_image = clip_transform(image)
        normalized_image = train_transform(image)
        return raw_image, normalized_image
    except Exception as e:
        print(f"Error loading image {file_name}: {e}")
        return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224)

class ProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(output_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
    def forward(self, x):
        return self.projection(x)

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(torch.device("cuda"))
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_projection = ProjectionLayer(512, 512).to(torch.device("cuda"))
text_projection = ProjectionLayer(512, 512).to(torch.device("cuda"))

def batch_extract_features(raw_images, captions, device):
    image_inputs = processor(images=[img for img in raw_images], return_tensors="pt", do_rescale=False)
    for key in image_inputs:
        image_inputs[key] = image_inputs[key].to(device)
    image_features = clip_model.get_image_features(**image_inputs)
    caption_tokens = processor(text=captions, padding=True, truncation=True, max_length=32, return_tensors="pt").to(device)
    text_features = clip_model.get_text_features(**caption_tokens)
    return image_features, text_features

def project_to_shared_space(image_features, text_features):
    return image_projection(image_features), text_projection(text_features)

def compute_similarity(image_features, text_features, temperature=0.05):
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    return torch.matmul(image_features, text_features.T) / temperature

class ImageCaptionDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = []
        for image_id, file_name, caption, label in pairs:
            raw_image, _ = load_image(file_name)
            self.pairs.append((raw_image, caption, label))
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        return self.pairs[idx]

train_dataset = ImageCaptionDataset(train_pairs)
val_dataset = ImageCaptionDataset(val_pairs)
test_dataset = ImageCaptionDataset(test_pairs)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

optimizer = optim.AdamW(
    list(image_projection.parameters()) + list(text_projection.parameters()),
    lr=1e-5,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = torch.amp.GradScaler(enabled=True)

patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(50):
    clip_model.train()
    image_projection.train()
    text_projection.train()
    total_loss = 0.0
    for raw_images, captions, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        optimizer.zero_grad()
        raw_images = raw_images.to(device)
        with torch.amp.autocast('cuda'):
            image_features, text_features = batch_extract_features(raw_images, captions, device)
            projected_image, projected_text = project_to_shared_space(image_features, text_features)
            similarity = compute_similarity(projected_image, projected_text)
            labels_tensor = torch.arange(similarity.size(0)).to(device)
            loss = loss_fn(similarity, labels_tensor)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    
    val_loss = 0.0
    clip_model.eval()
    image_projection.eval()
    text_projection.eval()
    with torch.no_grad():
        for raw_images, captions, labels in val_loader:
            raw_images = raw_images.to(device)
            image_features, text_features = batch_extract_features(raw_images, captions, device)
            projected_image, projected_text = project_to_shared_space(image_features, text_features)
            similarity = compute_similarity(projected_image, projected_text)
            labels_tensor = torch.arange(similarity.size(0)).to(device)
            loss = loss_fn(similarity, labels_tensor)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    
    scheduler.step()
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

torch.save(image_projection.state_dict(), "models/image_projection_phase1.pth")
torch.save(text_projection.state_dict(), "models/text_projection_phase1.pth")

def evaluate(loader):
    correct = 0
    total = 0
    clip_model.eval()
    image_projection.eval()
    text_projection.eval()
    with torch.no_grad():
        for raw_images, captions, labels in loader:
            raw_images = raw_images.to(device)
            image_features, text_features = batch_extract_features(raw_images, captions, device)
            projected_image, projected_text = project_to_shared_space(image_features, text_features)
            similarity = compute_similarity(projected_image, projected_text)
            _, predicted = torch.max(similarity, 1)
            total += labels.size(0)
            correct += (predicted == torch.arange(similarity.size(0)).to(device)).sum().item()
    return correct / total

train_accuracy = evaluate(train_loader)
val_accuracy = evaluate(val_loader)
test_accuracy = evaluate(test_loader)
print(f"Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}, Test Accuracy: {test_accuracy}")