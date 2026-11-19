from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image


# =====================================================
# CONFIGURATION
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_ID = "menes000/armocromia-12-season"

MANIFEST_PATH = (
    ROOT_DIR / "data" / "processed" / "splits"
    / "armocromia_split_seed42.csv"
)

CHECKPOINT_PATH = (
    ROOT_DIR / "models" / "checkpoints"
    / "colormate_cnn_baseline.pt"
)

OUTPUT_DIR = (
    ROOT_DIR / "reports" / "dataset_audit"
    / "confidence_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
NUM_WORKERS = 0
NUM_BINS = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================================
# MODEL ARCHITECTURE
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


# =====================================================
# PREPROCESSING
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# =====================================================
# DATASET
# =====================================================

class ColorMateDataset(Dataset):
    def __init__(self, hf_dataset, manifest_df, transform=None):
        self.hf_dataset = hf_dataset

        self.indices = (
            manifest_df["image_index"].astype(int).tolist()
        )

        self.labels = (
            manifest_df["label_id"].astype(int).tolist()
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


# =====================================================
# EXPECTED CALIBRATION ERROR
# =====================================================

def calculate_ece(confidences, correct, num_bins=10):
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)

    ece = 0.0
    bin_results = []

    for i in range(num_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower)
                & (confidences <= upper)
            )

        count = int(mask.sum())

        if count == 0:
            bin_results.append({
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "count": 0,
                "mean_confidence": np.nan,
                "accuracy": np.nan,
                "gap": np.nan
            })
            continue

        mean_conf = float(confidences[mask].mean())
        accuracy = float(correct[mask].mean())
        gap = abs(mean_conf - accuracy)

        ece += (count / len(confidences)) * gap

        bin_results.append({
            "bin": i + 1,
            "lower": lower,
            "upper": upper,
            "count": count,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
            "gap": gap
        })

    return ece, pd.DataFrame(bin_results)


# =====================================================
# LOAD CHECKPOINT AND DATA
# =====================================================

print("=" * 65)
print("COLORMATE - PREDICTION CONFIDENCE ANALYSIS")
print("=" * 65)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(MANIFEST_PATH)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

class_names = checkpoint["class_names"]
num_classes = checkpoint["num_classes"]

manifest = pd.read_csv(MANIFEST_PATH)

val_manifest = manifest[
    manifest["split"] == "validation"
].copy()

if len(val_manifest) != 760:
    raise ValueError(
        f"Expected 760 validation images, got {len(val_manifest)}"
    )

for _, row in val_manifest.iterrows():
    expected_name = class_names[int(row["label_id"])]

    if row["label"] != expected_name:
        raise ValueError(
            f"Label mismatch: {row['label']} != {expected_name}"
        )

print("\nCheckpoint epoch:", checkpoint["epoch"])
print("Validation images:", len(val_manifest))
print("Classes:", num_classes)

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


# =====================================================
# LOAD MODEL
# =====================================================

model = ColorMateCNN(num_classes).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nModel loaded successfully.")


# =====================================================
# PREDICTION AND CONFIDENCE
# =====================================================

all_targets = []
all_predictions = []
all_confidences = []
all_entropies = []

print("\nCalculating prediction probabilities...")

with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(DEVICE)

        logits = model(images)

        probabilities = torch.softmax(logits, dim=1)

        confidences, predictions = probabilities.max(dim=1)

        entropy = -(
            probabilities
            * torch.log(probabilities.clamp_min(1e-12))
        ).sum(dim=1)

        all_targets.extend(labels.numpy().tolist())

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_confidences.extend(
            confidences.cpu().numpy().tolist()
        )

        all_entropies.extend(
            entropy.cpu().numpy().tolist()
        )


# =====================================================
# BUILD PREDICTION DATAFRAME
# =====================================================

results = val_manifest.copy().reset_index(drop=True)

results["predicted_label_id"] = all_predictions

results["predicted_label"] = [
    class_names[p]
    for p in all_predictions
]

results["confidence"] = all_confidences

results["confidence_percent"] = (
    results["confidence"] * 100
)

results["entropy"] = all_entropies

results["correct"] = (
    results["label_id"]
    == results["predicted_label_id"]
)

prediction_path = OUTPUT_DIR / "validation_confidence_predictions.csv"

results.to_csv(prediction_path, index=False)


# =====================================================
# OVERALL CONFIDENCE SUMMARY
# =====================================================

correct_conf = results.loc[
    results["correct"], "confidence"
]

incorrect_conf = results.loc[
    ~results["correct"], "confidence"
]

print("\n" + "=" * 65)
print("OVERALL CONFIDENCE SUMMARY")
print("=" * 65)

print(f"Correct predictions: {len(correct_conf)}")
print(f"Incorrect predictions: {len(incorrect_conf)}")

print(
    f"\nMean confidence (correct): "
    f"{correct_conf.mean() * 100:.2f}%"
)

print(
    f"Mean confidence (incorrect): "
    f"{incorrect_conf.mean() * 100:.2f}%"
)

print(
    f"Median confidence (correct): "
    f"{correct_conf.median() * 100:.2f}%"
)

print(
    f"Median confidence (incorrect): "
    f"{incorrect_conf.median() * 100:.2f}%"
)


# =====================================================
# PER-CLASS CONFIDENCE
# =====================================================

class_summary = (
    results.groupby("label")
    .agg(
        sample_count=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_confidence=("confidence", "mean"),
        mean_entropy=("entropy", "mean")
    )
    .reindex(class_names)
    .reset_index()
    .rename(columns={"index": "label"})
)

class_summary["accuracy_percent"] = (
    class_summary["accuracy"] * 100
)

class_summary["confidence_percent"] = (
    class_summary["mean_confidence"] * 100
)

class_summary.to_csv(
    OUTPUT_DIR / "confidence_by_class.csv",
    index=False
)

print("\nCONFIDENCE BY CLASS")

print(
    class_summary[
        [
            "label",
            "sample_count",
            "accuracy_percent",
            "confidence_percent",
            "mean_entropy"
        ]
    ].to_string(index=False)
)


# =====================================================
# HIGH-CONFIDENCE ERRORS
# =====================================================

high_conf_errors = results[
    (~results["correct"])
    & (results["confidence"] >= 0.80)
].copy()

high_conf_errors = high_conf_errors.sort_values(
    "confidence",
    ascending=False
)

high_conf_errors.to_csv(
    OUTPUT_DIR / "high_confidence_errors.csv",
    index=False
)

print("\nHIGH-CONFIDENCE ERRORS (>= 80%)")
print("Count:", len(high_conf_errors))

if len(high_conf_errors) > 0:
    print(
        high_conf_errors[
            [
                "image_index",
                "label",
                "predicted_label",
                "confidence_percent"
            ]
        ].head(15).to_string(index=False)
    )


# =====================================================
# CALIBRATION ANALYSIS
# =====================================================

confidence_array = results["confidence"].to_numpy()

correct_array = results["correct"].to_numpy(dtype=float)

ece, calibration_df = calculate_ece(
    confidence_array,
    correct_array,
    NUM_BINS
)

calibration_df.to_csv(
    OUTPUT_DIR / "confidence_calibration_bins.csv",
    index=False
)

print("\nCALIBRATION SUMMARY")
print(f"Expected Calibration Error (ECE): {ece:.4f}")


# =====================================================
# CONFIDENCE DISTRIBUTION PLOT
# =====================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(
    correct_conf,
    bins=15,
    alpha=0.6,
    label="Correct",
    density=True
)

ax.hist(
    incorrect_conf,
    bins=15,
    alpha=0.6,
    label="Incorrect",
    density=True
)

ax.set_title(
    "ColorMate CNN - Confidence Distribution"
)

ax.set_xlabel("Prediction Confidence")
ax.set_ylabel("Density")

ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "confidence_distribution.png",
    dpi=160
)

plt.close(fig)


# =====================================================
# RELIABILITY DIAGRAM
# =====================================================

valid_bins = calibration_df[
    calibration_df["count"] > 0
]

fig, ax = plt.subplots(figsize=(7, 7))

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

ax.plot(
    valid_bins["mean_confidence"],
    valid_bins["accuracy"],
    marker="o",
    label="ColorMate CNN"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.set_xlabel("Mean Confidence")
ax.set_ylabel("Observed Accuracy")

ax.set_title("ColorMate CNN - Reliability Diagram")

ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "reliability_diagram.png",
    dpi=160
)

plt.close(fig)


# =====================================================
# FINAL OUTPUT
# =====================================================

print("\n" + "=" * 65)
print("CONFIDENCE ANALYSIS COMPLETED")
print("=" * 65)

print("\nGenerated files:")

for path in sorted(OUTPUT_DIR.iterdir()):
    if path.is_file():
        print(path)

print("\nAnalysis completed successfully.")