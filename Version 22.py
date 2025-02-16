import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import random
from PIL import Image
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch.optim as optim
from transformers import AutoModel, RobertaTokenizer, RobertaModel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, ColorJitter, RandomRotation
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np

def load_dataset(json_file):
    with open(json_file, "r") as f:
        return json.load(f)

def create_positive_negative_pairs(data, num_neg=5):
    image_to_captions = {item["file_name"]: item["captions"] for item in data}
    flat_data = [(img, cap) for img, caps in image_to_captions.items() for cap in caps]
    pos_pairs = [{"image": img, "caption": cap, "label": 1} for img, cap in flat_data]
    neg_pairs = [{"image": img, "caption": random.choice(flat_data)[1], "label": 0} 
                 for img, _ in flat_data for _ in range(num_neg) if img != random.choice(flat_data)[0]]
    return pos_pairs, neg_pairs

def get_augmentation_pipeline():
    return Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(0.5),
        ColorJitter(0.2, 0.2, 0.2, 0.1),
        RandomRotation(10),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_image(file_name, transform):
    try:
        image_path = f"data/images/{file_name}"
        image = Image.open(image_path).convert("RGB")
        return transform(image)
    except Exception as e:
        print(f"Error loading image {file_name}: {e}")
        return torch.zeros(3, 224, 224)

def encode_image(image, model, device):
    image_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        return model(image_tensor).last_hidden_state[:, 0].cpu().numpy()

def encode_text(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        return model(**inputs).last_hidden_state[:, 0].cpu().numpy()

class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(torch.nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.projection = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.projection(x)

class ContrastiveModel(torch.nn.Module):
    def __init__(self, image_embedding_dim, text_embedding_dim, shared_latent_dim):
        super().__init__()
        self.image_projection = ProjectionHead(input_dim=image_embedding_dim, output_dim=shared_latent_dim)
        self.text_projection = ProjectionHead(input_dim=text_embedding_dim, output_dim=shared_latent_dim)
    
    def forward(self, image_embeddings, text_embeddings):
        return self.image_projection(image_embeddings), self.text_projection(text_embeddings)

def contrastive_loss(image_embeddings, text_embeddings, temperature=0.07):
    logits = torch.matmul(image_embeddings, text_embeddings.T) / temperature
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i = torch.nn.CrossEntropyLoss()(logits, labels)
    loss_t = torch.nn.CrossEntropyLoss()(logits.T, labels)
    return (loss_i + loss_t) / 2

def train_model(pairs, image_folder, text_folder, epochs=10, batch_size=16, learning_rate=1e-4, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_embedding_dim = 1024
    text_embedding_dim = 768
    shared_latent_dim = 256
    model = ContrastiveModel(image_embedding_dim, text_embedding_dim, shared_latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    best_val_loss = float('inf')
    patience_counter = 0
    random.shuffle(pairs)
    split_idx = int(len(pairs) * 0.8)
    train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            image_embeddings = torch.stack([torch.tensor(np.load(os.path.join(image_folder, f"{os.path.splitext(pair['image'])[0]}.npy"))) for pair in batch]).to(device)
            text_embeddings = torch.stack([torch.tensor(np.load(os.path.join(text_folder, f"{os.path.splitext(pair['image'])[0]}_caption_{pair['caption']}.npy"))) for pair in batch]).to(device)
            optimizer.zero_grad()
            proj_image, proj_text = model(image_embeddings, text_embeddings)
            loss = contrastive_loss(proj_image, proj_text)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_pairs)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_pairs), batch_size):
                batch = val_pairs[i:i+batch_size]
                image_embeddings = torch.stack([torch.tensor(np.load(os.path.join(image_folder, f"{os.path.splitext(pair['image'])[0]}.npy"))) for pair in batch]).to(device)
                text_embeddings = torch.stack([torch.tensor(np.load(os.path.join(text_folder, f"{os.path.splitext(pair['image'])[0]}_caption_{pair['caption']}.npy"))) for pair in batch]).to(device)
                proj_image, proj_text = model(image_embeddings, text_embeddings)
                loss = contrastive_loss(proj_image, proj_text)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_pairs)
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_contrastive_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

if __name__ == "__main__":
    data = load_dataset("data/captions.json")
    pos_pairs, neg_pairs = create_positive_negative_pairs(data, num_neg=5)
    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)
    augmentation = get_augmentation_pipeline()
    os.makedirs("data/augmented_images", exist_ok=True)
    for img_file in tqdm([f for f in os.listdir("data/images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]):
        augmentation(Image.open(os.path.join("data/images", img_file)).convert("RGB"))\
            .save(os.path.join("data/augmented_images", f"{os.path.splitext(img_file)[0]}_aug.jpg"))
    dino_model = AutoModel.from_pretrained("facebook/dinov2-large").eval().to("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([Resize((224, 224)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image_folder = "data/augmented_images"
    embedding_folder = "data/embeddings"
    os.makedirs(embedding_folder, exist_ok=True)
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            embedding = encode_image(transform(Image.open(image_path).convert("RGB")), dino_model, "cuda" if torch.cuda.is_available() else "cpu")
            if embedding is not None:
                embedding_path = os.path.join(embedding_folder, f"{os.path.splitext(image_name)[0]}.npy")
                np.save(embedding_path, embedding)
    tokenizer, roberta_model = RobertaTokenizer.from_pretrained("roberta-base"), RobertaModel.from_pretrained("roberta-base").eval().to("cuda" if torch.cuda.is_available() else "cpu")
    txt_file, out_folder = "data/captions.json", "data/text_embeddings"
    os.makedirs(out_folder, exist_ok=True)
    with open(txt_file, "r") as f:
        data = json.load(f)
    for item in data:
        img_name = os.path.splitext(item["file_name"])[0]
        for i, cap in enumerate(item["captions"]):
            np.save(os.path.join(out_folder, f"{img_name}_caption_{i}.npy"), encode_text(cap, tokenizer, roberta_model, "cuda" if torch.cuda.is_available() else "cpu"))
    train_model(pairs, "data/embeddings", "data/text_embeddings")