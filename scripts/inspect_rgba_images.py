from datasets import load_dataset
from pathlib import Path

DATASET_NAME = "menes000/armocromia-12-season"

dataset = load_dataset(DATASET_NAME)

output_dir = Path("reports/dataset_audit/rgba_samples")
output_dir.mkdir(parents=True, exist_ok=True)

print("\nINSPECTING RGBA IMAGES")
print("=" * 50)

rgba_count = 0

for split_name, split_data in dataset.items():
    for index, record in enumerate(split_data):
        image = record["image"]

        if image.mode == "RGBA":
            label_id = record["label"]
            label_name = split_data.features["label"].names[label_id]

            output_path = (
                output_dir / f"{split_name}_{index}_{label_name}.png"
            )

            image.save(output_path)
            rgba_count += 1

            print(f"\nSplit: {split_name}")
            print(f"Index: {index}")
            print(f"Label: {label_name}")
            print(f"Mode: {image.mode}")
            print(f"Saved to: {output_path}")

print("\n" + "=" * 50)
print("Total RGBA images found:", rgba_count)
print("RGBA INSPECTION COMPLETED")