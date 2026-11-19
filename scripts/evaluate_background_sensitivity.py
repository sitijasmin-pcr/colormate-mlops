from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "menes000/armocromia-12-season"

CHECKPOINT_PATH = (
    ROOT / "models/checkpoints/colormate_cnn_baseline.pt"
)

SAMPLES_PATH = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "background_test_samples.csv"
)

SPLIT_PATH = (
    ROOT / "data/processed/splits/armocromia_split_seed42.csv"
)

PERTURBED_DIR = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "perturbed_images"
)

OUTPUT_DIR = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "cnn_sensitivity"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CONDITIONS = [
    "original",
    "gray_t50",
    "white_t50",
    "gray_t70",
    "white_t70",
]

print("=" * 65)
print("COLORMATE CNN BACKGROUND SENSITIVITY EVALUATION")
print("=" * 65)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =====================================================
# 2. VALIDATE FILES
# =====================================================

for path in [
    CHECKPOINT_PATH,
    SAMPLES_PATH,
    SPLIT_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

samples = pd.read_csv(SAMPLES_PATH)
split_manifest = pd.read_csv(SPLIT_PATH)

if len(samples) != 60:
    raise ValueError(
        f"Diharapkan 60 sampel, ditemukan {len(samples)}."
    )

if samples["image_index"].duplicated().any():
    raise ValueError("Ada image_index duplikat pada sampel.")

if samples["label"].isna().any():
    raise ValueError("Ada label kosong pada sampel.")

# Pastikan semua sampel berasal dari validation split.
validation_indices = set(
    split_manifest.loc[
        split_manifest["split"] == "validation",
        "image_index"
    ].astype(int)
)

sample_indices = set(
    samples["image_index"].astype(int)
)

if not sample_indices.issubset(validation_indices):
    raise ValueError(
        "Ada sampel yang bukan berasal dari validation split."
    )

print("\nValidation split check: PASSED")
print("Jumlah sampel:", len(samples))


# =====================================================
# 3. LOAD DATASET
# =====================================================

print("\nLoading Hugging Face dataset...")

dataset = load_dataset(DATASET_ID)
hf_train = dataset["train"]

class_names = hf_train.features["label"].names
num_classes = len(class_names)

print("Jumlah kelas:", num_classes)
print("Class names:", class_names)


# =====================================================
# 4. IMAGE TRANSFORM
#    HARUS SAMA DENGAN TRAINING
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
# 5. CNN ARCHITECTURE
#    IDENTIK DENGAN train_baseline.py
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


# =====================================================
# 6. LOAD CHECKPOINT
# =====================================================

print("\nLoading CNN checkpoint...")

# Checkpoint ini adalah file buatan project sendiri.
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)

if checkpoint["num_classes"] != num_classes:
    raise ValueError("Jumlah kelas checkpoint tidak cocok.")

if checkpoint["class_names"] != class_names:
    raise ValueError(
        "Urutan class_names checkpoint berbeda dari dataset."
    )

model = ColorMateCNN(num_classes).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True,
)

model.eval()

print("Checkpoint loaded successfully.")
print("Checkpoint epoch:", checkpoint["epoch"])
print("Validation accuracy saved:", checkpoint["val_accuracy"])
print("Validation macro F1 saved:", checkpoint["val_macro_f1"])

print(
    "Model parameters:",
    sum(p.numel() for p in model.parameters())
)


# =====================================================
# 7. IMAGE LOADING
# =====================================================

def load_condition_image(image_index, condition):

    if condition == "original":

        image = (
            hf_train[image_index]["image"]
            .convert("RGB")
        )

    else:

        image_path = (
            PERTURBED_DIR
            / condition
            / f"image_{image_index}.png"
        )

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image = Image.open(image_path).convert("RGB")

    return transform(image)


# =====================================================
# 8. INFERENCE
# =====================================================

@torch.inference_mode()
def predict_condition(image_indices, condition):

    all_probabilities = []

    for start in range(0, len(image_indices), BATCH_SIZE):

        batch_indices = image_indices[
            start:start + BATCH_SIZE
        ]

        images = [
            load_condition_image(index, condition)
            for index in batch_indices
        ]

        batch = torch.stack(images).to(DEVICE)

        logits = model(batch)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    return np.concatenate(all_probabilities, axis=0)


# =====================================================
# 9. RUN ALL CONDITIONS
# =====================================================

image_indices = samples["image_index"].astype(int).tolist()

true_labels = []

for index in image_indices:
    true_labels.append(
        int(hf_train[index]["label"])
    )

true_labels = np.array(true_labels)

# Pastikan label dari sampel dan dataset sesuai.
for position, row in samples.reset_index(drop=True).iterrows():

    dataset_label_name = class_names[true_labels[position]]

    if str(row["label"]) != dataset_label_name:
        raise ValueError(
            f"Label mismatch pada index {row['image_index']}: "
            f"{row['label']} vs {dataset_label_name}"
        )

predictions = {}

print("\nStarting inference...")

for condition in CONDITIONS:

    print(
        f"\nEvaluating condition: {condition}"
    )

    probabilities = predict_condition(
        image_indices,
        condition
    )

    if probabilities.shape != (60, num_classes):
        raise ValueError(
            f"Output shape tidak sesuai: {probabilities.shape}"
        )

    predictions[condition] = probabilities

    print("Completed:", condition)


# =====================================================
# 10. PER-IMAGE RESULTS
# =====================================================

original_probabilities = predictions["original"]

original_classes = original_probabilities.argmax(axis=1)

records = []

for condition in CONDITIONS:

    probabilities = predictions[condition]

    predicted_ids = probabilities.argmax(axis=1)

    confidences = probabilities.max(axis=1)

    for position, image_index in enumerate(image_indices):

        true_id = int(true_labels[position])
        predicted_id = int(predicted_ids[position])

        record = {
            "image_index": image_index,
            "true_label": class_names[true_id],
            "condition": condition,
            "predicted_label": class_names[predicted_id],
            "confidence": float(confidences[position]),
            "correct": bool(predicted_id == true_id),
            "prediction_changed_from_original": bool(
                predicted_id != original_classes[position]
            ),
            "original_prediction": class_names[
                int(original_classes[position])
            ],
            "original_confidence": float(
                original_probabilities[position].max()
            ),
            "confidence_change_from_original": float(
                confidences[position]
                - original_probabilities[position].max()
            ),
            "probability_mae_from_original": float(
                np.mean(
                    np.abs(
                        probabilities[position]
                        - original_probabilities[position]
                    )
                )
            ),
        }

        # Simpan probabilitas setiap kelas.
        for class_id, class_name in enumerate(class_names):
            record[
                f"prob_{class_name}"
            ] = float(probabilities[position, class_id])

        records.append(record)


results_df = pd.DataFrame(records)

results_path = (
    OUTPUT_DIR / "background_sensitivity_predictions.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig",
)


# =====================================================
# 11. CONDITION SUMMARY
# =====================================================

summary_records = []

for condition in CONDITIONS:

    subset = results_df[
        results_df["condition"] == condition
    ]

    y_pred = np.array([
        class_names.index(value)
        for value in subset["predicted_label"]
    ])

    accuracy = accuracy_score(
        true_labels,
        y_pred
    )

    macro_f1 = f1_score(
        true_labels,
        y_pred,
        average="macro",
        zero_division=0
    )

    changed = int(
        subset["prediction_changed_from_original"].sum()
    )

    summary_records.append({
        "condition": condition,
        "samples": len(subset),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "prediction_consistency_pct": float(
            (1 - changed / len(subset)) * 100
        ),
        "prediction_changed_count": changed,
        "mean_confidence": float(
            subset["confidence"].mean()
        ),
        "mean_confidence_change": float(
            subset["confidence_change_from_original"].mean()
        ),
        "mean_probability_mae": float(
            subset["probability_mae_from_original"].mean()
        ),
    })


summary_df = pd.DataFrame(summary_records)

summary_path = (
    OUTPUT_DIR / "background_sensitivity_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)


# =====================================================
# 12. CLASS-LEVEL SENSITIVITY
# =====================================================

class_records = []

for condition in CONDITIONS:

    subset = results_df[
        results_df["condition"] == condition
    ]

    for class_name in class_names:

        class_subset = subset[
            subset["true_label"] == class_name
        ]

        if len(class_subset) == 0:
            continue

        class_records.append({
            "condition": condition,
            "true_class": class_name,
            "samples": len(class_subset),
            "accuracy": float(
                class_subset["correct"].mean()
            ),
            "prediction_changed_count": int(
                class_subset[
                    "prediction_changed_from_original"
                ].sum()
            ),
            "mean_confidence": float(
                class_subset["confidence"].mean()
            ),
            "mean_probability_mae": float(
                class_subset[
                    "probability_mae_from_original"
                ].mean()
            ),
        })


class_df = pd.DataFrame(class_records)

class_path = (
    OUTPUT_DIR / "background_sensitivity_by_class.csv"
)

class_df.to_csv(
    class_path,
    index=False,
    encoding="utf-8-sig",
)


# =====================================================
# 13. PRINT SUMMARY
# =====================================================

print("\n" + "=" * 75)
print("BACKGROUND SENSITIVITY RESULTS")
print("=" * 75)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\nOriginal prediction comparison:")

print(
    results_df.groupby("condition")[
        "prediction_changed_from_original"
    ].sum()
)

print("\nSaved files:")
print(results_path)
print(summary_path)
print(class_path)

print("\n" + "=" * 75)
print("EVALUATION COMPLETED")
print("=" * 75)