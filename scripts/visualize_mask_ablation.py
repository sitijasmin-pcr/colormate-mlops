from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "mask_ablation"
)

SAMPLE_IDS = [3050, 2263, 2005, 2961, 2157, 13]

dataset = load_dataset("menes000/armocromia-12-season")
train_dataset = dataset["train"]

manifest = pd.read_csv(
    OUTPUT_DIR / "mask_ablation_manifest.csv"
)

for idx in SAMPLE_IDS:

    original = train_dataset[idx]["image"].convert("RGB")

    fig, axes = plt.subplots(
        1, 7,
        figsize=(21, 4)
    )

    axes[0].imshow(original)
    axes[0].set_title("Original")

    column = 1

    for kernel in [1, 3, 7]:
        for background in ["gray", "white"]:

            image_path = (
                OUTPUT_DIR
                / f"kernel_{kernel}"
                / background
                / f"image_{idx}.png"
            )

            image = Image.open(image_path).convert("RGB")

            axes[column].imshow(image)
            axes[column].set_title(
                f"{background.title()} K{kernel}"
            )

            column += 1

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Mask Ablation - Image {idx}",
        fontsize=14
    )

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"visual_{idx}.png"

    plt.savefig(
        output_path,
        dpi=130,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)

print("\nVISUAL MASK ABLATION COMPLETED")