from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


# =====================================================
# CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "menes000/armocromia-12-season"

RANDOM_SEED = 42
VALIDATION_SIZE = 0.20

OUTPUT_DIR = ROOT / "data" / "processed" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "armocromia_split_seed42.csv"


# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("COLORMATE - STRATIFIED DATA SPLIT")
print("=" * 60)

dataset = load_dataset(DATASET_ID)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

class_names = train_dataset.features["label"].names

labels = train_dataset["label"]
indices = list(range(len(train_dataset)))


# =====================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# =====================================================

train_indices, val_indices = train_test_split(
    indices,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels,
)

records = []

for index in train_indices:
    label_id = labels[index]

    records.append({
        "split": "train",
        "image_index": index,
        "label_id": label_id,
        "label": class_names[label_id],
    })

for index in val_indices:
    label_id = labels[index]

    records.append({
        "split": "validation",
        "image_index": index,
        "label_id": label_id,
        "label": class_names[label_id],
    })

# Preserve the original test split
test_labels = test_dataset["label"]

for index, label_id in enumerate(test_labels):
    records.append({
        "split": "test",
        "image_index": index,
        "label_id": label_id,
        "label": class_names[label_id],
    })


# =====================================================
# SAVE SPLIT MANIFEST
# =====================================================

df = pd.DataFrame(records)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


# =====================================================
# VALIDATION
# =====================================================

print("\nDataset split summary:")

print(
    df.groupby("split")
    .size()
    .to_string()
)

print("\nClass distribution:")

summary = pd.crosstab(
    df["label"],
    df["split"],
)

print(summary.to_string())

# Check no train-validation overlap
assert set(train_indices).isdisjoint(val_indices)

# Check all original training indices are preserved
assert len(train_indices) + len(val_indices) == len(labels)

# Check all classes exist in all splits
assert set(df[df["split"] == "train"]["label"]) == set(class_names)
assert set(df[df["split"] == "validation"]["label"]) == set(class_names)
assert set(df[df["split"] == "test"]["label"]) == set(class_names)

print("\nAll split validations passed!")
print("\nSaved manifest:")
print(OUTPUT_FILE)