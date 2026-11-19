from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = ROOT / "models/checkpoints/colormate_cnn_baseline.pt"

SAMPLES_PATH = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "background_test_samples.csv"
)

MASK_DIR = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "mask_ablation"
)

OUTPUT_DIR = MASK_DIR / "cnn_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONDITIONS = [
    "original",
    "gray_k1", "white_k1",
    "gray_k3", "white_k3",
    "gray_k7", "white_k7",
]


# =====================================================
# MODEL ARCHITECTURE - HARUS SAMA DENGAN BASELINE
# =====================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
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

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# =====================================================
# VALIDATION DAN CHECKPOINT
# =====================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)

samples = pd.read_csv(SAMPLES_PATH)

if len(samples) != 60 or samples["image_index"].nunique() != 60:
    raise ValueError("Sampel harus terdiri dari 60 index unik.")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

class_names = checkpoint["class_names"]

model = ColorMateCNN(len(class_names)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Device:", DEVICE)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Classes:", len(class_names))


# =====================================================
# DATASET DAN TRANSFORM
# =====================================================

print("\nLoading Hugging Face dataset...")

dataset = load_dataset("menes000/armocromia-12-season")
train_dataset = dataset["train"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


for _, row in samples.iterrows():
    idx = int(row["image_index"])

    if idx < 0 or idx >= len(train_dataset):
        raise ValueError(f"Invalid dataset index: {idx}")

    actual_label_id = train_dataset[idx]["label"]

    # Hugging Face ClassLabel biasanya menghasilkan integer.
    if isinstance(actual_label_id, (int, np.integer)):
        actual_label_name = class_names[int(actual_label_id)]
    else:
        actual_label_name = str(actual_label_id)

    expected_label = str(row["label"])

    if actual_label_name != expected_label:
        raise ValueError(
            f"Label mismatch at index {idx}: "
            f"dataset={actual_label_name}, CSV={expected_label}"
        )

print("Dataset label validation: PASSED")



# =====================================================
# LOAD IMAGE BERDASARKAN KONDISI
# =====================================================

def load_condition_image(idx, condition):

    if condition == "original":
        return train_dataset[idx]["image"].convert("RGB")

    background, kernel = condition.split("_")

    image_path = (
        MASK_DIR
        / f"kernel_{kernel[1:]}"
        / background
        / f"image_{idx}.png"
    )

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    return Image.open(image_path).convert("RGB")


# =====================================================
# INFERENCE
# =====================================================

all_results = []

with torch.inference_mode():

    for condition in CONDITIONS:

        print(f"\nEvaluating: {condition}")

        for start in range(0, len(samples), BATCH_SIZE):

            batch_rows = samples.iloc[start:start + BATCH_SIZE]

            images = []
            labels = []
            indices = []

            for _, row in batch_rows.iterrows():

                idx = int(row["image_index"])
                label = str(row["label"])

                image = load_condition_image(idx, condition)

                images.append(transform(image))
                labels.append(class_names.index(label))
                indices.append(idx)

            batch = torch.stack(images).to(DEVICE)

            logits = model(batch)
            probabilities = torch.softmax(logits, dim=1)

            confidences, predictions = probabilities.max(dim=1)

            predictions = predictions.cpu().numpy()
            confidences = confidences.cpu().numpy()
            probabilities = probabilities.cpu().numpy()

            for i, idx in enumerate(indices):

                record = {
                    "condition": condition,
                    "image_index": idx,
                    "true_label": class_names[labels[i]],
                    "predicted_label": class_names[predictions[i]],
                    "confidence": float(confidences[i]),
                    "correct": bool(predictions[i] == labels[i]),
                }

                for class_index, class_name in enumerate(class_names):
                    record[f"prob_{class_name}"] = float(
                        probabilities[i, class_index]
                    )

                all_results.append(record)

        print("Completed:", condition)


# =====================================================
# SUMMARY
# =====================================================

results = pd.DataFrame(all_results)

original = (
    results[results["condition"] == "original"]
    [["image_index", "predicted_label", "confidence"]]
    .rename(columns={
        "predicted_label": "original_prediction",
        "confidence": "original_confidence",
    })
)

results = results.merge(original, on="image_index", how="left")

results["prediction_changed"] = (
    results["predicted_label"] != results["original_prediction"]
)

results["confidence_change"] = (
    results["confidence"] - results["original_confidence"]
)

probability_columns = [f"prob_{name}" for name in class_names]

original_probabilities = (
    results[results["condition"] == "original"]
    [["image_index"] + probability_columns]
    .set_index("image_index")
)

mae_values = []

for _, row in results.iterrows():

    original_probs = original_probabilities.loc[
        row["image_index"], probability_columns
    ].to_numpy(dtype=float)

    current_probs = row[probability_columns].to_numpy(dtype=float)

    mae_values.append(
        float(np.mean(np.abs(current_probs - original_probs)))
    )

results["probability_mae_vs_original"] = mae_values

summary_records = []

for condition in CONDITIONS:

    subset = results[results["condition"] == condition]

    summary_records.append({
        "condition": condition,
        "samples": len(subset),
        "accuracy": accuracy_score(
            subset["true_label"],
            subset["predicted_label"]
        ),
        "macro_f1": f1_score(
            subset["true_label"],
            subset["predicted_label"],
            labels=class_names,
            average="macro",
            zero_division=0,
        ),
        "prediction_consistency_pct": (
            100 * (1 - subset["prediction_changed"].mean())
        ),
        "prediction_changed_count": int(
            subset["prediction_changed"].sum()
        ),
        "mean_confidence": subset["confidence"].mean(),
        "mean_confidence_change": subset["confidence_change"].mean(),
        "mean_probability_mae": (
            subset["probability_mae_vs_original"].mean()
        ),
    })

summary = pd.DataFrame(summary_records)

results.to_csv(
    OUTPUT_DIR / "mask_ablation_predictions.csv",
    index=False
)

summary.to_csv(
    OUTPUT_DIR / "mask_ablation_summary.csv",
    index=False
)

print("\n" + "=" * 75)
print("MASK ABLATION CNN EVALUATION")
print("=" * 75)

print(summary.round(4).to_string(index=False))

print("\nSaved:")
print(OUTPUT_DIR / "mask_ablation_predictions.csv")
print(OUTPUT_DIR / "mask_ablation_summary.csv")

print("\nEVALUATION COMPLETED")