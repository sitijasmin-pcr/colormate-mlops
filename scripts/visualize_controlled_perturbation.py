from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

SAMPLES_CSV = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "background_test_samples.csv"
)

CONTROLLED_DIR = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "controlled_perturbed_images"
)

OUTPUT_DIR = (
    CONTROLLED_DIR / "contact_sheets"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

samples = pd.read_csv(SAMPLES_CSV)

dataset = load_dataset("menes000/armocromia-12-season")
train_dataset = dataset["train"]

ROWS_PER_PAGE = 10

for page, start in enumerate(
    range(0, len(samples), ROWS_PER_PAGE),
    start=1
):

    selected = samples.iloc[
        start:start + ROWS_PER_PAGE
    ]

    fig, axes = plt.subplots(
        len(selected),
        3,
        figsize=(12, 3 * len(selected)),
        squeeze=False
    )

    for row_number, (_, row) in enumerate(selected.iterrows()):

        image_index = int(row["image_index"])
        label = str(row["label"])

        original = (
            train_dataset[image_index]["image"]
            .convert("RGB")
        )

        gray_path = (
            CONTROLLED_DIR / "gray"
            / f"image_{image_index}.png"
        )

        white_path = (
            CONTROLLED_DIR / "white"
            / f"image_{image_index}.png"
        )

        gray = Image.open(gray_path).convert("RGB")
        white = Image.open(white_path).convert("RGB")

        axes[row_number, 0].imshow(original)
        axes[row_number, 1].imshow(gray)
        axes[row_number, 2].imshow(white)

        axes[row_number, 0].set_ylabel(
            f"{image_index}\n{label}",
            fontsize=8
        )

        for ax in axes[row_number]:
            ax.axis("off")

    titles = [
        "Original",
        "Controlled Gray",
        "Controlled White"
    ]

    for col, title in enumerate(titles):
        axes[0, col].set_title(
            title,
            fontsize=11,
            fontweight="bold"
        )

    fig.suptitle(
        f"Controlled Background Audit - Page {page}",
        fontsize=14
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"controlled_contact_sheet_{page}.png"
    )

    plt.savefig(
        output_path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)

print("\nCONTROLLED VISUAL AUDIT COMPLETED")