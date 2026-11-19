from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from datasets import load_dataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "data/external/models/segmentation/"
    "selfie_multiclass_256x256.tflite"
)

SAMPLES_PATH = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "background_test_samples.csv"
)

OUTPUT_DIR = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "mask_ablation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_SIZES = [1, 3, 7]

BACKGROUNDS = {
    "gray": np.array([128, 128, 128], dtype=np.uint8),
    "white": np.array([255, 255, 255], dtype=np.uint8),
}

FOREGROUND_CLASSES = [1, 2, 3, 4, 5]

samples = pd.read_csv(SAMPLES_PATH)

if len(samples) != 60:
    raise ValueError(f"Expected 60 samples, got {len(samples)}")

dataset = load_dataset("menes000/armocromia-12-season")
train_dataset = dataset["train"]

options = mp.tasks.vision.ImageSegmenterOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    output_category_mask=True,
    output_confidence_masks=False,
)

records = []

with mp.tasks.vision.ImageSegmenter.create_from_options(
    options
) as segmenter:

    for position, row in samples.iterrows():

        idx = int(row["image_index"])
        label = str(row["label"])

        print(f"[{position + 1}/60] Image {idx} - {label}")

        original_image = (
            train_dataset[idx]["image"].convert("RGB")
        )
        original = np.asarray(original_image)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=original
        )

        result = segmenter.segment(mp_image)

        category = (
            result.category_mask
            .numpy_view()
            .squeeze()
            .astype(np.uint8)
        )

        raw_mask = np.isin(
            category,
            FOREGROUND_CLASSES
        ).astype(np.uint8)

        raw_mask_image = Image.fromarray(raw_mask * 255)

        raw_mask = (
            np.asarray(
                raw_mask_image.resize(
                    original_image.size,
                    Image.Resampling.NEAREST
                )
            ) > 0
        ).astype(np.uint8)

        for kernel_size in KERNEL_SIZES:

            kernel = np.ones(
                (kernel_size, kernel_size),
                dtype=np.uint8
            )

            if kernel_size == 1:
                protected_mask = raw_mask.copy()
            else:
                protected_mask = cv2.dilate(
                    raw_mask,
                    kernel,
                    iterations=1
                )

            foreground = protected_mask.astype(bool)

            record = {
                "image_index": idx,
                "label": label,
                "kernel_size": kernel_size,
                "raw_area_pct": float(raw_mask.mean() * 100),
                "protected_area_pct": float(foreground.mean() * 100),
            }

            for background_name, color in BACKGROUNDS.items():

                output = original.copy()
                output[~foreground] = color

                variant_dir = (
                    OUTPUT_DIR
                    / f"kernel_{kernel_size}"
                    / background_name
                )

                variant_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                output_path = (
                    variant_dir / f"image_{idx}.png"
                )

                Image.fromarray(output).save(output_path)

                # Verify exact preservation.
                difference = (
                    output.astype(np.int16)
                    - original.astype(np.int16)
                )

                preserved = not np.any(
                    difference[foreground] != 0
                )

                record[
                    f"{background_name}_preserved"
                ] = bool(preserved)

                record[
                    f"{background_name}_changed_pixels"
                ] = int(
                    np.any(difference != 0, axis=2).sum()
                )

                if not preserved:
                    raise AssertionError(
                        f"Foreground changed: {idx}"
                    )

            records.append(record)

manifest = pd.DataFrame(records)

manifest_path = OUTPUT_DIR / "mask_ablation_manifest.csv"
manifest.to_csv(manifest_path, index=False)

summary = (
    manifest.groupby("kernel_size")
    .agg(
        samples=("image_index", "nunique"),
        mean_raw_area=("raw_area_pct", "mean"),
        mean_protected_area=("protected_area_pct", "mean"),
        min_protected_area=("protected_area_pct", "min"),
        max_protected_area=("protected_area_pct", "max"),
    )
    .round(2)
)

summary_path = OUTPUT_DIR / "mask_ablation_summary.csv"
summary.to_csv(summary_path)

print("\n" + "=" * 55)
print("MASK ABLATION COMPLETED")
print("=" * 55)
print(summary.to_string())
print("\nManifest:", manifest_path)
print("Summary:", summary_path)
print("Output:", OUTPUT_DIR)