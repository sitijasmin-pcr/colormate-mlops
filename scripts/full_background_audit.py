from datasets import load_dataset
from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path

DATASET_NAME = "menes000/armocromia-12-season"

OUTPUT_DIR = Path("reports/dataset_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)
train_data = dataset["train"]

class_names = train_data.features["label"].names

results = []

print("\nStarting full background audit...")

for index in range(len(train_data)):

    sample = train_data[index]

    image = sample["image"].convert("RGB")
    array = np.asarray(image, dtype=np.uint8)

    # Near-black pixel ratio
    black_mask = np.all(array <= 5, axis=2)
    black_ratio = black_mask.mean() * 100

    # Border near-black ratio
    border = np.concatenate([
        array[:10, :, :].reshape(-1, 3),
        array[-10:, :, :].reshape(-1, 3),
        array[:, :10, :].reshape(-1, 3),
        array[:, -10:, :].reshape(-1, 3)
    ])

    border_black = np.all(border <= 5, axis=1)
    border_ratio = border_black.mean() * 100

    label_id = int(sample["label"])
    label_name = class_names[label_id]

    results.append({
        "index": index,
        "label": label_name,
        "black_ratio": black_ratio,
        "border_black_ratio": border_ratio
    })

    if (index + 1) % 500 == 0:
        print(f"Processed: {index + 1}/{len(train_data)}")


# ==========================================
# SAVE PER-IMAGE RESULTS
# ==========================================

df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "full_background_audit.csv"
df.to_csv(csv_path, index=False)


# ==========================================
# CLASS SUMMARY
# ==========================================

summary = df.groupby("label").agg(
    image_count=("index", "count"),
    black_mean=("black_ratio", "mean"),
    black_std=("black_ratio", "std"),
    black_min=("black_ratio", "min"),
    black_max=("black_ratio", "max"),
    border_mean=("border_black_ratio", "mean"),
    border_std=("border_black_ratio", "std")
).round(2)

print("\n" + "=" * 65)
print("FULL BACKGROUND AUDIT SUMMARY")
print("=" * 65)

print(summary.to_string())

print("\nOverall statistics:")
print(df[["black_ratio", "border_black_ratio"]].describe().round(2))

print("\nResults saved to:", csv_path)
print("\nFULL BACKGROUND AUDIT COMPLETED")