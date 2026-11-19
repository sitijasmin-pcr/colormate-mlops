from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from collections import Counter
from io import BytesIO

DATASET_NAME = "menes000/armocromia-12-season"

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

print("\n" + "=" * 55)
print("IMAGE QUALITY VALIDATION")
print("=" * 55)

for split_name, split_data in dataset.items():
    print(f"\nSPLIT: {split_name}")

    total = len(split_data)
    valid_images = 0
    corrupted_images = []
    dimensions = Counter()
    modes = Counter()
    formats = Counter()

    for index, record in enumerate(split_data):
        try:
            image = record["image"]

            if image is None:
                raise ValueError("Image is None")

            image.load()

            valid_images += 1
            dimensions[image.size] += 1
            modes[image.mode] += 1
            formats[image.format] += 1

        except Exception as error:
            corrupted_images.append({
                "index": index,
                "error": str(error)
            })

        if (index + 1) % 500 == 0:
            print(f"Checked: {index + 1}/{total}")

    print("\nRESULT:")
    print("Total images:", total)
    print("Valid images:", valid_images)
    print("Corrupted images:", len(corrupted_images))

    print("\nImage dimensions:")
    for dimension, count in dimensions.items():
        print(f"{dimension}: {count}")

    print("\nImage modes:")
    print(modes)

    print("\nImage formats:")
    print(formats)

    if corrupted_images:
        print("\nCorrupted image samples:")
        for item in corrupted_images[:10]:
            print(item)

print("\nIMAGE QUALITY VALIDATION COMPLETED")