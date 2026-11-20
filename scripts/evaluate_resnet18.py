import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "menes000/armocromia-12-season"

MANIFEST_PATH = (
    ROOT / "data" / "processed" / "splits"
    / "armocromia_split_seed42.csv"
)

CHECKPOINT_PATH = (
    ROOT / "models" / "checkpoints"
    / "colormate_resnet18_experiment01.pt"
)

OUTPUT_DIR = (
    ROOT / "reports" / "experiments"
    / "resnet18_experiment01" / "evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 0
SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================================
# VALIDATION TRANSFORM
# =====================================================

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


# =====================================================
# DATASET
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
            raise ValueError(
                f"Label mismatch pada image_index {index}"
            )

        if self.transform:
            image = self.transform(image)

        return image, label, index


# =====================================================
# LOAD CHECKPOINT
# =====================================================

print("=" * 65)
print("COLORMATE RESNET18 - EVALUATION")
print("=" * 65)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint tidak ditemukan: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]

if len(class_names) != num_classes:
    raise ValueError("Jumlah class_names tidak sesuai num_classes.")

print("\nCheckpoint:", CHECKPOINT_PATH)
print("Best epoch:", checkpoint["epoch"])
print("Checkpoint validation loss:", checkpoint["val_loss"])
print("Checkpoint validation accuracy:", checkpoint["val_accuracy"])
print("Checkpoint validation macro F1:", checkpoint["val_macro_f1"])


# =====================================================
# LOAD MANIFEST
# =====================================================

manifest = pd.read_csv(MANIFEST_PATH)

required_columns = {
    "split",
    "image_index",
    "label_id",
    "label",
}

if not required_columns.issubset(manifest.columns):
    raise ValueError("Kolom manifest tidak lengkap.")

val_manifest = manifest[
    manifest["split"] == "validation"
].copy()

if len(val_manifest) != 760:
    raise ValueError(
        f"Jumlah validation tidak sesuai: {len(val_manifest)}"
    )

if val_manifest["image_index"].duplicated().any():
    raise ValueError("Ditemukan image_index duplikat pada validation.")

for _, row in val_manifest.iterrows():
    label_id = int(row["label_id"])

    if label_id < 0 or label_id >= num_classes:
        raise ValueError(f"label_id tidak valid: {label_id}")

    if row["label"] != class_names[label_id]:
        raise ValueError(
            f"Label tidak sesuai pada image_index {row['image_index']}"
        )

print("\nValidation images:", len(val_manifest))


# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading Hugging Face dataset...")

dataset = load_dataset(DATASET_ID)

dataset_class_names = dataset["train"].features["label"].names

if dataset_class_names != class_names:
    raise ValueError(
        "Urutan kelas dataset berbeda dengan checkpoint."
    )

val_data = ColorMateDataset(
    dataset["train"],
    val_manifest,
    val_transform,
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
)


# =====================================================
# BUILD MODEL
# =====================================================

model = resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes,
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()


# =====================================================
# INFERENCE
# =====================================================

print("\nRunning evaluation...")

all_labels = []
all_predictions = []
all_confidences = []
all_indices = []

with torch.inference_mode():

    for images, labels, indices in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_confidences.extend(confidences.cpu().numpy().tolist())
        all_indices.extend(indices.numpy().tolist())


# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions,
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0,
)

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
)

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(
    OUTPUT_DIR / "classification_report.csv",
    index=True,
)

predictions_df = pd.DataFrame({
    "image_index": all_indices,
    "true_label_id": all_labels,
    "true_label": [class_names[x] for x in all_labels],
    "predicted_label_id": all_predictions,
    "predicted_label": [class_names[x] for x in all_predictions],
    "confidence": all_confidences,
})

predictions_df["correct"] = (
    predictions_df["true_label_id"]
    == predictions_df["predicted_label_id"]
)

predictions_df.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False,
)


# =====================================================
# CONFUSION MATRIX
# =====================================================

fig, ax = plt.subplots(figsize=(14, 12))

image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(image, ax=ax)

ax.set(
    xticks=np.arange(num_classes),
    yticks=np.arange(num_classes),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel="Predicted Label",
    ylabel="True Label",
    title="ResNet18 - Validation Confusion Matrix",
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right",
    rotation_mode="anchor",
)

threshold = cm.max() / 2 if cm.max() > 0 else 0

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "black",
            fontsize=8,
        )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# =====================================================
# TOP CONFUSION PAIRS
# =====================================================

confusion_pairs = []

for true_id in range(num_classes):
    for pred_id in range(num_classes):

        if true_id == pred_id:
            continue

        count = int(cm[true_id, pred_id])

        if count > 0:
            confusion_pairs.append({
                "true_label": class_names[true_id],
                "predicted_label": class_names[pred_id],
                "count": count,
            })

confusion_pairs = sorted(
    confusion_pairs,
    key=lambda x: x["count"],
    reverse=True,
)

confusion_df = pd.DataFrame(confusion_pairs)

confusion_df.to_csv(
    OUTPUT_DIR / "top_confusion_pairs.csv",
    index=False,
)


# =====================================================
# SUMMARY
# =====================================================

summary = {
    "experiment": "ResNet18 Experiment 01",
    "checkpoint_epoch": int(checkpoint["epoch"]),
    "validation_samples": len(all_labels),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "weighted_f1": float(weighted_f1),
    "checkpoint_val_loss": float(checkpoint["val_loss"]),
    "checkpoint_val_accuracy": float(checkpoint["val_accuracy"]),
    "checkpoint_val_macro_f1": float(checkpoint["val_macro_f1"]),
    "class_names": class_names,
    "output_directory": str(OUTPUT_DIR),
}

with open(
    OUTPUT_DIR / "evaluation_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4)


# =====================================================
# RESULTS
# =====================================================

print("\n" + "=" * 65)
print("EVALUATION COMPLETED")
print("=" * 65)

print("Checkpoint epoch:", checkpoint["epoch"])
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"Validation Macro F1: {macro_f1:.4f}")
print(f"Validation Weighted F1: {weighted_f1:.4f}")

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )
)

print("\nTOP CONFUSION PAIRS")

if confusion_df.empty:
    print("Tidak ada kesalahan klasifikasi.")
else:
    print(confusion_df.head(10).to_string(index=False))

print("\nOutput directory:", OUTPUT_DIR)
print("Confusion matrix: confusion_matrix.png")
print("Classification report: classification_report.csv")
print("Predictions: validation_predictions.csv")
print("Confusion pairs: top_confusion_pairs.csv")
print("Summary: evaluation_summary.json")
