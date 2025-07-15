"""
Cassava Leaf Disease – End‑to‑End CNN Example
=============================================

This single Python script is laid out like a notebook: execute it **cell‑by‑cell** in
Jupyter/Colab, or run the whole file with `python cassava_nn_example.py` after editing
the paths and hyper‑parameters section below.

What it does
------------
*   Loads the Cassava Leaf Disease Classification **train.zip** extracted from Kaggle –
    a folder of JPG images (`train_images/`) and a `train.csv` with labels.
*   Builds a *very small* convolutional neural network in **PyTorch** (or you can switch
    to a pretrained ResNet‑18 with one line).
*   Trains for a few epochs, prints loss/accuracy curves, and saves the best model.
*   Shows how to predict on **one arbitrary JPG** so you can test any image you have.

Requirements
------------
```bash
pip install torch torchvision torchaudio pandas numpy pillow matplotlib scikit‑learn
```

Usage – quick start
-------------------
1. Download the Kaggle dataset and unzip so you have:
   ```
   data/
     cassava/
       train_images/  ← all .jpg files
       train.csv      ← image_id,label pairs
   ```
2. Edit `DATA_DIR` in the first code cell to point at this folder.
3. Run the script.  GPU is used automatically if available.

Hyper‑parameters, data splits, and model depth are all defined near the top so you can
experiment easily.
"""

# %% ─────────────────────────── 1. Imports & Config ─────────────────────────────
from pathlib import Path
import random
from typing import Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
DATA_DIR = Path("data/cassava")  # ← CHANGE ME if your data lives elsewhere
CSV_PATH = DATA_DIR / "train.csv"
IMAGES_DIR = DATA_DIR / "train_images"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Hyper‑parameters ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
NUM_CLASSES = 5        # Cassava has 5 disease classes
IMAGE_SIZE = 224       # images will be center‑cropped to 224×224
BATCH_SIZE = 32
NUM_EPOCHS = 10        # bump this up once everything works
LR = 3e‑4
TRAIN_VAL_SPLIT = 0.8  # 80‑20 train/val split

# %% ─────────────────────── 2. Dataset & Transforms ─────────────────────────––
class CassavaDataset(Dataset):
    """Custom `Dataset` for Cassava Leaf images."""

    def __init__(self, csv_file: Path, img_dir: Path, transform: T.Compose):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.img_dir / row.image_id
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row.label)
        return image, label

# Image augmentations (train) and basic preprocess (val/test)
train_tfms = T.Compose([
    T.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(20),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_tfms = T.Compose([
    T.Resize(IMAGE_SIZE + 32),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Build full dataset then split
full_ds = CassavaDataset(CSV_PATH, IMAGES_DIR, transform=train_tfms)
train_len = int(TRAIN_VAL_SPLIT * len(full_ds))
val_len = len(full_ds) - train_len
train_ds, val_ds = random_split(full_ds, [train_len, val_len])

# IMPORTANT: Validation must use val transforms (no augmentation)
train_ds.dataset.transform = train_tfms
val_ds.dataset.transform = val_tfms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# %% ─────────────────────────── 3. Model Definition ─────────────────────────––
class TinyCNN(nn.Module):
    """A *very* small CNN for quick experiments."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMAGE_SIZE // 8) * (IMAGE_SIZE // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# To switch to a ResNet‑18 backbone:
# model = models.resnet18(weights="IMAGENET1K_V1")
# model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = TinyCNN(NUM_CLASSES).to(DEVICE)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# %% ─────────────────────── 4. Training & Validation ─────────────────────────––

def epoch_loop(loader: DataLoader, train: bool = False):
    """Runs one epoch. Returns mean loss and accuracy."""
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss, preds_list, labels_list = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        with torch.set_grad_enabled(train):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
        preds_list.append(outputs.argmax(1).detach().cpu().numpy())
        labels_list.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(preds_list)
    y_true = np.concatenate(labels_list)
    acc = accuracy_score(y_true, y_pred)
    return epoch_loss / len(loader.dataset), acc

train_history, val_history = [], []
best_val_acc, best_epoch = 0.0, -1

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss, train_acc = epoch_loop(train_loader, train=True)
    val_loss, val_acc = epoch_loop(val_loader, train=False)
    train_history.append((train_loss, train_acc))
    val_history.append((val_loss, val_acc))

    print(f"Epoch {epoch:02}/{NUM_EPOCHS} | "
          f"Train L: {train_loss:.4f} A: {train_acc:.3f} | "
          f"Val L: {val_loss:.4f} A: {val_acc:.3f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc, best_epoch = val_acc, epoch
        torch.save(model.state_dict(), MODEL_DIR / "best_cassava_cnn.pth")

print(f"Best Val Acc: {best_val_acc:.3f} at epoch {best_epoch}")

# %% ───────────────────────── 5. Plot Learning Curves ─────────────────────────––
plt.figure(figsize=(8, 4))
plt.plot([x[1] for x in train_history], label="Train Acc")
plt.plot([x[1] for x in val_history], label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Learning Curve"); plt.legend()
plt.show()

# %% ─────────────────────────── 6. Inference Demo ─────────────────────────–––
@torch.inference_mode()
def predict_image(img_path: Path) -> int:
    """Returns the integer class for a single image path."""
    img = Image.open(img_path).convert("RGB")
    img_t = val_tfms(img).unsqueeze(0).to(DEVICE)
    outputs = model(img_t)
    return outputs.argmax(1).item()

# Example: change to any jpg you like
# sample_prediction = predict_image(Path("my_leaf.jpg"))
# print("Predicted class: ", sample_prediction)

# %% ──────────────────────────── 7. Confusion Matrix ─────────────────────────––
all_preds, all_labels = [], []
model.load_state_dict(torch.load(MODEL_DIR / "best_cassava_cnn.pth", map_location=DEVICE))
model.eval()
for imgs, labels in val_loader:
    imgs = imgs.to(DEVICE)
    outputs = model(imgs)
    all_preds.append(outputs.argmax(1).cpu().numpy())
    all_labels.append(labels.numpy())
cm = confusion_matrix(np.concatenate(all_labels), np.concatenate(all_preds))
print("Confusion Matrix:\n", cm)

# %% ───────────────────────── End of Script ─────────────────────────────────––
if __name__ == "__main__":
    print("✔ Training complete. Best model saved to", MODEL_DIR / "best_cassava_cnn.pth")
