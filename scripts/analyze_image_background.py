from datasets import load_dataset
from collections import Counter
import numpy as np

DATASET_NAME = "menes000/armocromia-12-season"
SAMPLE_PER_CLASS = 10
RANDOM_SEED = 42

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

train_data = dataset["train"]
class_names = train_data.features["label"].names

rng = np.random.default_rng(RANDOM_SEED)

print("\n" + "=" * 55)
print("COLORMATE BACKGROUND AND PIXEL AUDIT")
print("=" * 55)

total_samples = 0

for class_id, class_name in enumerate(class_names):

    indices = [
        i for i, label in enumerate(train_data["label"])
        if label == class_id
    ]

    selected = rng.choice(
        indices,
        size=min(SAMPLE_PER_CLASS, len(indices)),
        replace=False
    )

    black_ratios = []
    border_black_ratios = []
    mean_colors = []

    for index in selected:

        image = train_data[int(index)]["image"].convert("RGB")
        array = np.asarray(image, dtype=np.uint8)

        # Pixel yang seluruh kanal RGB-nya bernilai <= 5
        black_pixels = np.all(array <= 5, axis=2)

        black_ratio = black_pixels.mean() * 100

        # Periksa border gambar setebal 10 pixel
        border = np.concatenate([
            array[:10, :, :].reshape(-1, 3),
            array[-10:, :, :].reshape(-1, 3),
            array[:, :10, :].reshape(-1, 3),
            array[:, -10:, :].reshape(-1, 3)
        ])

        border_black = np.all(border <= 5, axis=1)
        border_ratio = border_black.mean() * 100

        black_ratios.append(black_ratio)
        border_black_ratios.append(border_ratio)
        mean_colors.append(array.mean(axis=(0, 1)))

        total_samples += 1

    print(f"\nCLASS: {class_name}")
    print(f"Samples: {len(selected)}")
    print(
        "Average near-black pixel ratio:",
        f"{np.mean(black_ratios):.2f}%"
    )
    print(
        "Average border near-black ratio:",
        f"{np.mean(border_black_ratios):.2f}%"
    )
    print(
        "Average RGB image value:",
        np.mean(mean_colors, axis=0).round(2)
    )

print("\n" + "=" * 55)
print("TOTAL SAMPLES:", total_samples)
print("BACKGROUND AUDIT COMPLETED")
print("=" * 55)