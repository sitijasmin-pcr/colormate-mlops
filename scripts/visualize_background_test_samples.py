from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "menes000/armocromia-12-season"

SAMPLE_PATH = (
    ROOT / "reports/dataset_audit/background_perturbation"
    / "background_test_samples.csv"
)

OUTPUT_PATH = (
    ROOT / "reports/dataset_audit/background_perturbation"
    / "background_test_contact_sheet.png"
)

samples = pd.read_csv(SAMPLE_PATH)

print("Loading Hugging Face dataset...")
dataset = load_dataset(DATASET_ID)["train"]

classes = samples["label"].drop_duplicates().tolist()

fig, axes = plt.subplots(
    12, 5,
    figsize=(15, 30)
)

for row, label in enumerate(classes):

    class_samples = samples[
        samples["label"] == label
    ].sort_values("image_index")

    for col, (_, sample) in enumerate(class_samples.iterrows()):

        image = dataset[int(sample["image_index"])]["image"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB")

        ax = axes[row, col]

        ax.imshow(image)
        ax.set_title(
            f"Index: {sample['image_index']}",
            fontsize=8
        )
        ax.axis("off")

    axes[row, 0].set_ylabel(
        label,
        fontsize=9,
        rotation=0,
        labelpad=65,
        va="center"
    )

fig.suptitle(
    "ColorMate - Background Perturbation Test Samples",
    fontsize=16,
    fontweight="bold"
)

fig.tight_layout(rect=[0.04, 0, 1, 0.98])

fig.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("\nTotal images:", len(samples))
print("Saved:", OUTPUT_PATH)