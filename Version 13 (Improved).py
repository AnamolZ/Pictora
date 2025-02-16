import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import random
from PIL import Image
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaModel, CLIPProcessor, CLIPModel
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

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_caption(caption, max_length=32):
    return tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)

class ProjectionLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(output_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.projection(x)

image_projection = ProjectionLayer(768, 512)
text_projection = ProjectionLayer(768, 512)

def batch_extract_features(raw_images, captions, device):
    image_inputs = processor(images=[img for img in raw_images], return_tensors="pt", do_rescale=False)
    for key in image_inputs:
        image_inputs[key] = image_inputs[key].to(device)
    image_features = clip_model.get_image_features(**image_inputs)
    caption_tokens = tokenizer(captions, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
    caption_tokens = {k: v.to(device) for k, v in caption_tokens.items()}
    text_outputs = text_encoder(**caption_tokens)
    text_features = text_outputs.last_hidden_state.mean(dim=1)
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_model_state = None
    def __call__(self, val_loss, model_state):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model_state
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
text_encoder.to(device)
image_projection.to(device)
text_projection.to(device)

if torch.cuda.device_count() > 1:
    clip_model = torch.nn.DataParallel(clip_model)
    text_encoder = torch.nn.DataParallel(text_encoder)
    image_projection = torch.nn.DataParallel(image_projection)
    text_projection = torch.nn.DataParallel(text_projection)

optimizer = optim.AdamW(
    list(image_projection.parameters()) + list(text_projection.parameters()),
    lr=1e-5,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
early_stopping = EarlyStopping(patience=5)

num_epochs = 50
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    clip_model.train()
    text_encoder.train()
    image_projection.train()
    text_projection.train()
    total_loss = 0.0
    for raw_images, captions, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
        optimizer.zero_grad()
        raw_images = raw_images.to(device)
        with torch.cuda.amp.autocast():
            image_features, text_features = batch_extract_features(raw_images, captions, device)
            projected_image, projected_text = project_to_shared_space(image_features, text_features)
            similarity = compute_similarity(projected_image, projected_text)
            labels_tensor = torch.arange(similarity.size(0)).to(device)
            loss = loss_fn(similarity, labels_tensor)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(list(image_projection.parameters()) + list(text_projection.parameters()), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    
    val_loss = 0.0
    clip_model.eval()
    text_encoder.eval()
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
    scheduler.step(avg_val_loss)
    
    state = {
        "clip_model": clip_model.state_dict(),
        "text_encoder": text_encoder.state_dict(),
        "image_projection": image_projection.state_dict(),
        "text_projection": text_projection.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    early_stopping(avg_val_loss, state)
    if early_stopping.early_stop:
        break

torch.save(early_stopping.best_model_state["image_projection"], "models/image_projection.pth")
torch.save(early_stopping.best_model_state["text_projection"], "models/text_projection.pth")

def evaluate(loader):
    correct = 0
    total = 0
    clip_model.eval()
    text_encoder.eval()
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