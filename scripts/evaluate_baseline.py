from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)


# =========================================================
# CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_ID = "menes000/armocromia-12-season"

MANIFEST_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "splits"
    / "armocromia_split_seed42.csv"
)

CHECKPOINT_PATH = (
    ROOT_DIR
    / "models"
    / "checkpoints"
    / "colormate_cnn_baseline.pt"
)

OUTPUT_DIR = ROOT_DIR / "reports" / "dataset_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# MODEL ARCHITECTURE
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
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
            ConvBlock(128, 256)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# =========================================================
# PREPROCESSING
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# =========================================================
# DATASET
# =========================================================

class ColorMateDataset(Dataset):
    def __init__(self, hf_dataset, manifest_df, transform=None):
        self.hf_dataset = hf_dataset

        self.indices = (
            manifest_df["image_index"]
            .astype(int)
            .tolist()
        )

        self.labels = (
            manifest_df["label_id"]
            .astype(int)
            .tolist()
        )

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_index = self.indices[idx]
        label = self.labels[idx]

        image = self.hf_dataset[image_index]["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# LOAD CHECKPOINT AND DATA
# =========================================================

print("=" * 65)
print("COLORMATE CNN BASELINE EVALUATION")
print("=" * 65)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(MANIFEST_PATH)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]

print("\nCheckpoint epoch:", checkpoint["epoch"])
print("Number of classes:", num_classes)
print("Checkpoint validation loss:", checkpoint["val_loss"])

manifest = pd.read_csv(MANIFEST_PATH)

val_manifest = manifest[
    manifest["split"] == "validation"
].copy()

if len(val_manifest) != 760:
    raise ValueError(
        f"Expected 760 validation images, got {len(val_manifest)}"
    )

# Pastikan label manifest sesuai urutan kelas checkpoint.
for _, row in val_manifest.iterrows():
    expected_name = class_names[int(row["label_id"])]

    if row["label"] != expected_name:
        raise ValueError(
            f"Label mismatch at image index {row['image_index']}: "
            f"{row['label']} != {expected_name}"
        )

print("\nLoading Hugging Face dataset...")
hf_dataset = load_dataset(DATASET_ID)

validation_dataset = ColorMateDataset(
    hf_dataset["train"],
    val_manifest,
    transform=transform
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda")
)


# =========================================================
# LOAD MODEL
# =========================================================

model = ColorMateCNN(num_classes=num_classes).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nModel loaded successfully.")


# =========================================================
# PREDICTION
# =========================================================

all_targets = []
all_predictions = []

print("\nRunning validation predictions...")

with torch.no_grad():
    for images, labels in validation_loader:

        images = images.to(DEVICE)
        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        all_targets.extend(labels.numpy().tolist())
        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    all_targets,
    all_predictions
)

macro_f1 = f1_score(
    all_targets,
    all_predictions,
    labels=list(range(num_classes)),
    average="macro",
    zero_division=0
)

report = classification_report(
    all_targets,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_path = OUTPUT_DIR / "baseline_classification_report.csv"
report_df.to_csv(report_path)

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=list(range(num_classes))
)

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_path = OUTPUT_DIR / "baseline_confusion_matrix.csv"
cm_df.to_csv(cm_path)


# =========================================================
# CONFUSION MATRIX VISUALIZATION
# =========================================================

fig, ax = plt.subplots(figsize=(13, 11))

image = ax.imshow(cm, interpolation="nearest", cmap="Blues")

ax.set_title("ColorMate CNN - Validation Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))

ax.set_xticklabels(
    class_names,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(class_names)

threshold = cm.max() / 2

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=8
        )

fig.colorbar(image, ax=ax)
fig.tight_layout()

cm_image_path = OUTPUT_DIR / "baseline_confusion_matrix.png"

plt.savefig(
    cm_image_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# SAVE PREDICTIONS
# =========================================================

prediction_df = val_manifest.copy()

prediction_df["predicted_label_id"] = all_predictions

prediction_df["predicted_label"] = [
    class_names[pred]
    for pred in all_predictions
]

prediction_df["correct"] = (
    prediction_df["label_id"]
    == prediction_df["predicted_label_id"]
)

prediction_path = OUTPUT_DIR / "baseline_validation_predictions.csv"

prediction_df.to_csv(
    prediction_path,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 65)
print("VALIDATION EVALUATION RESULTS")
print("=" * 65)

print("Validation images:", len(all_targets))
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro F1: {macro_f1:.4f} ({macro_f1 * 100:.2f}%)")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        all_targets,
        all_predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )
)

print("\nSaved files:")
print(report_path)
print(cm_path)
print(cm_image_path)
print(prediction_path)

print("\nEvaluation completed successfully.")