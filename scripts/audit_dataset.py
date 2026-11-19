from datasets import load_dataset

DATASET_NAME = "menes000/armocromia-12-season"

print("=" * 55)
print("COLORMATE - PUBLIC DATASET AUDIT")
print("=" * 55)

# Load dataset
dataset = load_dataset(DATASET_NAME)

print("\n[1] DATASET STRUCTURE")
print(dataset)

# Inspect each split
for split_name, split_data in dataset.items():
    print(f"\n[2] SPLIT: {split_name}")
    print("Total images:", len(split_data))

    label_feature = split_data.features["label"]
    class_names = label_feature.names

    print("\nClass distribution:")

    labels = split_data["label"]

    for class_id, class_name in enumerate(class_names):
        count = labels.count(class_id)
        print(f"{class_name:20s}: {count}")

    print("\nImage features:")
    print(split_data.features)
    
print("\n" + "=" * 55)
print("DATASET AUDIT COMPLETED")
print("=" * 55)