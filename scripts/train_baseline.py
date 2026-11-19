from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "menes000/armocromia-12-season"

MANIFEST_PATH = (
    ROOT / "data" / "processed" / "splits"
    / "armocromia_split_seed42.csv"
)

CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "colormate_cnn_baseline.pt"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
PATIENCE = 5
NUM_WORKERS = 0


# =====================================================
# REPRODUCIBILITY
# =====================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("COLORMATE CNN BASELINE TRAINING")
print("=" * 65)
print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================================
# LOAD DATA AND MANIFEST
# =====================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(MANIFEST_PATH)

manifest = pd.read_csv(MANIFEST_PATH)

print("\nLoading Hugging Face dataset...")
dataset = load_dataset(DATASET_ID)

class_names = dataset["train"].features["label"].names
num_classes = len(class_names)

print("Classes:", num_classes)
print("Class names:", class_names)


# =====================================================
# TRANSFORMS
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


# =====================================================
# PYTORCH DATASET
# =====================================================

class ColorMateDataset(Dataset):

    def __init__(self, hf_dataset, manifest_df, transform=None):
        self.hf_dataset = hf_dataset
        self.indices = manifest_df["image_index"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_index = self.indices[position]

        row = self.hf_dataset[original_index]

        image = row["image"].convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


# =====================================================
# CREATE TRAIN AND VALIDATION LOADERS
# =====================================================

train_manifest = manifest[
    manifest["split"] == "train"
].copy()

val_manifest = manifest[
    manifest["split"] == "validation"
].copy()

train_data = ColorMateDataset(
    dataset["train"],
    train_manifest,
    transform,
)

val_data = ColorMateDataset(
    dataset["train"],
    val_manifest,
    transform,
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)

print("\nTraining images:", len(train_data))
print("Validation images:", len(val_data))


# =====================================================
# CNN MODEL
# =====================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.block(x)


class ColorMateCNN(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


model = ColorMateCNN(num_classes).to(DEVICE)

print("\nModel parameters:",
      sum(p.numel() for p in model.parameters()))


# =====================================================
# LOSS AND OPTIMIZER
# =====================================================

# Class weights based on training split only.
class_counts = (
    train_manifest["label_id"]
    .value_counts()
    .reindex(range(num_classes), fill_value=0)
    .sort_index()
)

if (class_counts == 0).any():
    raise ValueError("A class is missing from training data.")

class_weights = (
    len(train_manifest) / (num_classes * class_counts)
)

class_weights = torch.tensor(
    class_weights.to_numpy(),
    dtype=torch.float32,
    device=DEVICE,
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# =====================================================
# TRAINING AND VALIDATION FUNCTIONS
# =====================================================

def train_one_epoch():

    model.train()

    total_loss = 0
    predictions = []
    targets = []

    for images, labels in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions.extend(
            outputs.argmax(dim=1).detach().cpu().tolist()
        )

        targets.extend(labels.detach().cpu().tolist())

    return (
        total_loss / len(train_data),
        accuracy_score(targets, predictions),
        f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
    )


@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0
    predictions = []
    targets = []

    for images, labels in val_loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions.extend(
            outputs.argmax(dim=1).cpu().tolist()
        )

        targets.extend(labels.cpu().tolist())

    return (
        total_loss / len(val_data),
        accuracy_score(targets, predictions),
        f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
    )


# =====================================================
# TRAINING LOOP
# =====================================================

best_val_loss = float("inf")
epochs_without_improvement = 0

history = []

for epoch in range(EPOCHS):

    train_loss, train_acc, train_f1 = train_one_epoch()

    val_loss, val_acc, val_f1 = validate()

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
    })

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train | Loss: {train_loss:.4f} "
        f"| Accuracy: {train_acc:.4f} "
        f"| Macro F1: {train_f1:.4f}"
    )

    print(
        f"Valid | Loss: {val_loss:.4f} "
        f"| Accuracy: {val_acc:.4f} "
        f"| Macro F1: {val_f1:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
            "seed": SEED,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        }, CHECKPOINT_PATH)

        print("Best checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# =====================================================
# SAVE TRAINING HISTORY
# =====================================================

history_path = (
    ROOT / "reports" / "dataset_audit"
    / "baseline_training_history.csv"
)

history_path.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame(history).to_csv(
    history_path,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 65)
print("TRAINING COMPLETED")
print("=" * 65)

print("Best validation loss:", best_val_loss)
print("Checkpoint:", CHECKPOINT_PATH)
print("History:", history_path)