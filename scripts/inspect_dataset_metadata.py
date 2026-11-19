from datasets import load_dataset
from collections import Counter

DATASET_NAME = "menes000/armocromia-12-season"

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

print("\n" + "=" * 50)
print("DATASET METADATA AUDIT")
print("=" * 50)

for split_name, split_data in dataset.items():

    print(f"\nSPLIT: {split_name}")
    print("Number of images:", len(split_data))

    print("\nDataset features:")
    print(split_data.features)

    print("\nAvailable columns:")
    print(split_data.column_names)

    print("\nExample record keys:")
    sample = split_data[0]
    print(list(sample.keys()))

    print("\nImage information:")
    image = sample["image"]

    print("Image format:", image.format)
    print("Image mode:", image.mode)
    print("Image dimensions:", image.size)

    labels = split_data["label"]
    class_names = split_data.features["label"].names

    counts = Counter(labels)

    print("\nClass distribution:")

    for class_id, class_name in enumerate(class_names):
        print(f"{class_name}: {counts[class_id]}")

print("\nMETADATA AUDIT COMPLETED")