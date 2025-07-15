"""
Cassava Leaf Disease – Tiny Dataset CNN Example
==============================================

This Python script demonstrates how to build and train a small CNN using only 15 images
from the Cassava Leaf Disease dataset. It's a simplified version suitable for practicing
deep learning workflows with very limited data.

What it does
------------
*   Loads 15 labeled JPG images from a folder.
*   Builds a minimal convolutional neural network in **PyTorch**.
*   Trains the model for a few epochs and shows accuracy trends.
*   Evaluates and predicts on test data.

Requirements
------------
```bash
pip install torch torchvision pandas numpy pillow matplotlib
```

Usage – quick start
-------------------
1. Prepare a folder with 15 labeled images and a CSV:
   ```
   data/
     small_cassava/
       images/         ← 15 .jpg files
       labels.csv      ← image_id,label pairs
   ```
2. Update `DATA_DIR` below to match the location of your data.
3. Run the script.
"""

# %% 1. Imports & Config
from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths and hyperparameters
DATA_DIR = Path("data/small_cassava")  # CHANGE ME
CSV_PATH = DATA_DIR / "labels.csv"
IMAGES_DIR = DATA_DIR / "images"
NUM_CLASSES = 5
IMAGE_SIZE = 128
BATCH_SIZE = 4
NUM_EPOCHS = 20
LR = 1e-3

# %% 2. Dataset & Transforms
class TinyCassavaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row.image_id).convert("RGB")
        img = self.transform(img)
        return img, int(row.label)

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

full_ds = TinyCassavaDataset(CSV_PATH, IMAGES_DIR, transform)
train_len = int(0.8 * len(full_ds))
val_len = len(full_ds) - train_len
train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# %% 3. Model
class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = SmallCNN(NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# %% 4. Training Loop
train_accs, val_accs = [], []

def evaluate(loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

for epoch in range(NUM_EPOCHS):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    train_acc = evaluate(train_loader)
    val_acc = evaluate(val_loader)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Acc: {train_acc:.2f} - Val Acc: {val_acc:.2f}")

# %% 5. Plot Accuracy
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# %% 6. Predict One Image
@torch.no_grad()
def predict_image(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    model.eval()
    pred = model(img).argmax(1).item()
    return pred

# Example:
# prediction = predict_image(IMAGES_DIR / "your_image.jpg")
# print("Predicted class:", prediction)
