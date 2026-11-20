from pathlib import Path
import random
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "menes000/armocromia-12-season"

MANIFEST_PATH = (
    ROOT / "data" / "processed" / "splits"
    / "armocromia_split_seed42.csv"
)

OUTPUT_DIR = ROOT / "reports" / "experiments" / "resnet18_experiment01"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = (
    ROOT / "models" / "checkpoints"
    / "colormate_resnet18_experiment01.pt"
)
CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
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
print("COLORMATE RESNET18 - EXPERIMENT 01")
print("=" * 65)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================================
# LOAD DATA
# =====================================================

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(MANIFEST_PATH)

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "image_index", "label_id", "label", "split"
}

if not required_columns.issubset(manifest.columns):
    raise ValueError("Manifest tidak memiliki kolom yang diperlukan.")

for split_name, split_df in manifest.groupby("split"):
    if split_df["image_index"].duplicated().any():
        raise ValueError(
            f"Ditemukan image_index duplikat pada split {split_name}."
        )

print("\nLoading dataset...")
dataset = load_dataset(DATASET_ID)

class_names = dataset["train"].features["label"].names
num_classes = len(class_names)

print("Classes:", class_names)

train_manifest = manifest[
    manifest["split"] == "train"
].copy()

val_manifest = manifest[
    manifest["split"] == "validation"
].copy()

if len(train_manifest) != 3038 or len(val_manifest) != 760:
    raise ValueError("Jumlah data train/validation tidak sesuai manifest.")

if set(train_manifest["image_index"]) & set(val_manifest["image_index"]):
    raise ValueError("Train dan validation mengalami overlap.")

for _, row in manifest.iterrows():
    expected_name = class_names[int(row["label_id"])]

    if row["label"] != expected_name:
        raise ValueError(
            f"Label mismatch pada index {row['image_index']}"
        )


# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

val_transform = transforms.Compose([
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
        self.labels = manifest_df["label_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        label = self.labels[position]

        row = self.hf_dataset[index]
        image = row["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        if int(row["label"]) != label:
            raise ValueError(f"Label dataset tidak cocok pada index {index}")

        if self.transform:
            image = self.transform(image)

        return image, label


train_data = ColorMateDataset(
    dataset["train"],
    train_manifest,
    train_transform,
)

val_data = ColorMateDataset(
    dataset["train"],
    val_manifest,
    val_transform,
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
# RESNET18 MODEL
# =====================================================

weights = ResNet18_Weights.DEFAULT

model = resnet18(weights=weights)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes,
)

model = model.to(DEVICE)

print("\nModel: ResNet18 ImageNet pretrained")
print(
    "Parameters:",
    sum(p.numel() for p in model.parameters())
)


# =====================================================
# CLASS WEIGHTS
# =====================================================

class_counts = (
    train_manifest["label_id"]
    .value_counts()
    .reindex(range(num_classes), fill_value=0)
    .sort_index()
)

if (class_counts == 0).any():
    raise ValueError("Ada kelas yang tidak memiliki data training.")

class_weights = (
    len(train_manifest) / (num_classes * class_counts)
)

class_weights = torch.tensor(
    class_weights.to_numpy(),
    dtype=torch.float32,
    device=DEVICE,
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# =====================================================
# TRAINING FUNCTION
# =====================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0
    predictions = []
    targets = []

    for images, labels in train_loader:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

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
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        ),
    )


# =====================================================
# VALIDATION FUNCTION
# =====================================================

@torch.no_grad()
def validate():

    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    for images, labels in val_loader:

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

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
            labels=list(range(num_classes)),
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

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

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
            "experiment": "resnet18_experiment01",
            "architecture": "resnet18",
            "pretrained": "ImageNet1K_V1",
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "class_names": class_names,
            "num_classes": num_classes,
            "seed": SEED,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "config": {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "augmentation": [
                    "horizontal_flip_0.5",
                    "rotation_5_degrees",
                ],
            },
        }, CHECKPOINT_PATH)

        print("Best checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# =====================================================
# SAVE RESULTS
# =====================================================

history_path = OUTPUT_DIR / "training_history.csv"

pd.DataFrame(history).to_csv(
    history_path,
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "experiment": "ResNet18 Experiment 01",
    "best_val_loss": best_val_loss,
    "best_epoch": int(
        pd.DataFrame(history).loc[
            pd.DataFrame(history)["val_loss"].idxmin(),
            "epoch"
        ]
    ),
    "checkpoint": str(CHECKPOINT_PATH),
    "history": str(history_path),
}

with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)

print("\n" + "=" * 65)
print("RESNET18 EXPERIMENT COMPLETED")
print("=" * 65)

print("Best validation loss:", best_val_loss)
print("Checkpoint:", CHECKPOINT_PATH)
print("History:", history_path)
print("Summary:", OUTPUT_DIR / "summary.json")
