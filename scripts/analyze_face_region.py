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

print("\nAnalyzing non-black image regions...")

for index in range(len(train_data)):

    sample = train_data[index]

    image = sample["image"].convert("RGB")
    array = np.asarray(image)

    # Pixel dianggap non-black jika setidaknya
    # satu kanal RGB memiliki nilai lebih dari 5.
    non_black = np.any(array > 5, axis=2)

    ys, xs = np.where(non_black)

    if len(xs) == 0:
        continue

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    bbox_area_ratio = (
        bbox_width * bbox_height
        / (224 * 224)
    ) * 100

    non_black_ratio = non_black.mean() * 100

    label_id = int(sample["label"])

    results.append({
        "index": index,
        "label": class_names[label_id],
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area_ratio": bbox_area_ratio,
        "non_black_ratio": non_black_ratio
    })

    if (index + 1) % 500 == 0:
        print(f"Processed: {index + 1}/{len(train_data)}")


df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "face_region_audit.csv"
df.to_csv(csv_path, index=False)

summary = df.groupby("label").agg(
    image_count=("index", "count"),
    bbox_area_mean=("bbox_area_ratio", "mean"),
    bbox_area_std=("bbox_area_ratio", "std"),
    bbox_width_mean=("bbox_width", "mean"),
    bbox_height_mean=("bbox_height", "mean"),
    non_black_mean=("non_black_ratio", "mean")
).round(2)

print("\n" + "=" * 65)
print("FACE REGION AUDIT SUMMARY")
print("=" * 65)

print(summary.to_string())

print("\nOverall statistics:")
print(
    df[[
        "bbox_area_ratio",
        "non_black_ratio"
    ]].describe().round(2)
)

print("\nResults saved to:", csv_path)
print("\nFACE REGION AUDIT COMPLETED")