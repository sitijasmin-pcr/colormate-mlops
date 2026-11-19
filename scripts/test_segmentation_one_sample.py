from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "data/external/models/segmentation/"
    "selfie_multiclass_256x256.tflite"
)

SAMPLES_CSV = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "background_test_samples.csv"
)

OUTPUT_PATH = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "segmentation_first_sample.png"
)

# Ambil satu sampel pertama dari daftar 60 gambar
samples = pd.read_csv(SAMPLES_CSV)
sample = samples.iloc[0]
image_index = int(sample["image_index"])

print("Memuat dataset...")
dataset = load_dataset("menes000/armocromia-12-season")

original_image = dataset["train"][image_index]["image"].convert("RGB")

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=np.asarray(original_image)
)

options = mp.tasks.vision.ImageSegmenterOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    output_category_mask=True,
    output_confidence_masks=True,
)

print("Menjalankan segmentasi...")

with mp.tasks.vision.ImageSegmenter.create_from_options(options) as segmenter:
    result = segmenter.segment(mp_image)

print("Model labels:", segmenter.labels)
print("\nHASIL SEGMENTASI")
print("Image index:", image_index)
print("Label dataset:", sample["label"])
print("Jumlah confidence masks:", len(result.confidence_masks or []))

fig, axes = plt.subplots(
    1,
    2 + len(result.confidence_masks or []),
    figsize=(4 * (2 + len(result.confidence_masks or [])), 5)
)

axes[0].imshow(original_image)
axes[0].set_title("Original")

if result.category_mask is not None:
    category = result.category_mask.numpy_view()
    print("Category mask shape:", category.shape)
    print("Category values:", np.unique(category))

    axes[1].imshow(category, cmap="tab10")
    axes[1].set_title("Category Mask")
else:
    axes[1].set_title("No Category Mask")

for i, mask in enumerate(result.confidence_masks or []):
    mask_array = mask.numpy_view()

    print(
        f"Confidence mask {i}:",
        "shape =", mask_array.shape,
        "min =", float(mask_array.min()),
        "max =", float(mask_array.max())
    )

    axes[i + 2].imshow(mask_array, cmap="viridis", vmin=0, vmax=1)
    axes[i + 2].set_title(f"Confidence {i}")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
plt.close()

print("\nGambar hasil:", OUTPUT_PATH)
print("Tes satu sampel selesai.")
