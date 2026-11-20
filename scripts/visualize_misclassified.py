from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from PIL import Image


# =====================================================
# CONFIGURATION
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    ROOT_DIR
    / "reports"
    / "dataset_audit"
    / "baseline_validation_predictions.csv"
)

OUTPUT_DIR = ROOT_DIR / "reports" / "dataset_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "menes000/armocromia-12-season"

MAX_IMAGES_PER_PAIR = 5

TARGET_PAIRS = [
    ("autumn_deep", "winter_deep"),
    ("winter_bright", "winter_deep"),
    ("spring_light", "summer_light"),
    ("summer_soft", "summer_light"),
    ("spring_warm", "autumn_warm"),
]


# =====================================================
# LOAD PREDICTIONS
# =====================================================

if not PREDICTION_PATH.exists():
    raise FileNotFoundError(PREDICTION_PATH)

df = pd.read_csv(PREDICTION_PATH)

required_columns = [
    "image_index",
    "label",
    "predicted_label",
    "confidence",
    "correct",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

if len(df) != 760:
    raise ValueError(
        f"Expected 760 validation samples, got {len(df)}"
    )

print("=" * 65)
print("COLORMATE - MISCLASSIFIED IMAGE VISUALIZATION")
print("=" * 65)

print("Validation images:", len(df))
print("Incorrect predictions:", (~df["correct"].astype(bool)).sum())


# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading Hugging Face dataset...")

dataset = load_dataset(DATASET_ID)
hf_train = dataset["train"]


# =====================================================
# SELECT MISCLASSIFIED SAMPLES
# =====================================================

selected_samples = []

for true_label, predicted_label in TARGET_PAIRS:

    pair_data = df[
        (df["label"] == true_label)
        & (df["predicted_label"] == predicted_label)
        & (~df["correct"].astype(bool))
    ].copy()

    pair_data = pair_data.sort_values(
        "confidence",
        ascending=False
    )

    selected = pair_data.head(MAX_IMAGES_PER_PAIR)

    print(
        f"\n{true_label} -> {predicted_label}: "
        f"{len(pair_data)} errors, "
        f"{len(selected)} selected"
    )

    for _, row in selected.iterrows():

        selected_samples.append({
            "true_label": true_label,
            "predicted_label": predicted_label,
            "image_index": int(row["image_index"]),
            "confidence": float(row["confidence"]),
        })


if not selected_samples:
    raise ValueError(
        "No matching misclassified samples found."
    )


# =====================================================
# CREATE CONTACT SHEET
# =====================================================

n_cols = MAX_IMAGES_PER_PAIR
n_rows = len(TARGET_PAIRS)

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(19, 20)
)

axes = axes.reshape(n_rows, n_cols)

fig.suptitle(
    "ColorMate CNN - Misclassified Validation Images",
    fontsize=17,
    fontweight="bold"
)

sample_position = 0

for row_idx, (true_label, predicted_label) in enumerate(TARGET_PAIRS):

    pair_samples = [
        sample for sample in selected_samples
        if sample["true_label"] == true_label
        and sample["predicted_label"] == predicted_label
    ]

    for col_idx in range(n_cols):

        ax = axes[row_idx, col_idx]
        ax.axis("off")

        if col_idx >= len(pair_samples):
            continue

        sample = pair_samples[col_idx]

        image_index = sample["image_index"]

        image = hf_train[image_index]["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        ax.imshow(image)

        ax.set_title(
            f"True: {sample['true_label']}\n"
            f"Pred: {sample['predicted_label']}\n"
            f"Confidence: {sample['confidence']:.2%}\n"
            f"Index: {image_index}",
            fontsize=8
        )

        sample_position += 1

    axes[row_idx, 0].set_ylabel(
        f"{true_label}\n↓\n{predicted_label}",
        fontsize=10,
        rotation=0,
        labelpad=85,
        va="center"
    )


plt.tight_layout(rect=[0.04, 0, 1, 0.96])

output_path = (
    OUTPUT_DIR
    / "baseline_misclassified_contact_sheet.png"
)

plt.savefig(
    output_path,
    dpi=160,
    bbox_inches="tight"
)

plt.close(fig)


# =====================================================
# SUMMARY
# =====================================================

print("\nTotal images visualized:", sample_position)
print("Contact sheet saved:")
print(output_path)

print("\nVisualization completed successfully.")