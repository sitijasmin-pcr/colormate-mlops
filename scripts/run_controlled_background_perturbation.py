from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from datasets import load_dataset
from PIL import Image


# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "data/external/models/segmentation/"
    "selfie_multiclass_256x256.tflite"
)

SAMPLES_CSV = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "background_test_samples.csv"
)

OUTPUT_DIR = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "controlled_perturbed_images"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BACKGROUNDS = {
    "gray": np.array([128, 128, 128], dtype=np.uint8),
    "white": np.array([255, 255, 255], dtype=np.uint8),
}

# MediaPipe category IDs:
# 0 background
# 1 hair
# 2 body-skin
# 3 face-skin
# 4 clothes
# 5 others/accessories

FOREGROUND_CLASSES = [1, 2, 3, 4, 5]

# Dilation protects nearby foreground boundaries.
DILATION_KERNEL_SIZE = 7


# =====================================================
# 2. VALIDATE INPUTS
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)

if not SAMPLES_CSV.exists():
    raise FileNotFoundError(SAMPLES_CSV)

samples = pd.read_csv(SAMPLES_CSV)

if len(samples) != 60:
    raise ValueError(
        f"Expected 60 samples, got {len(samples)}"
    )

print("Samples:", len(samples))
print("Dilation kernel:", DILATION_KERNEL_SIZE)


# =====================================================
# 3. LOAD DATASET
# =====================================================

print("\nLoading Hugging Face dataset...")

dataset = load_dataset("menes000/armocromia-12-season")
train_dataset = dataset["train"]


# =====================================================
# 4. CREATE CATEGORY MASK
# =====================================================

def create_preservation_mask(result, original_size):
    """
    Creates a binary foreground mask from category IDs.

    Resize uses nearest-neighbor interpolation to preserve
    categorical class IDs.
    """

    if result.category_mask is None:
        raise RuntimeError("Category mask unavailable.")

    category = (
        result.category_mask
        .numpy_view()
        .squeeze()
        .astype(np.uint8)
    )

    # Preserve all foreground classes.
    mask = np.isin(
        category,
        FOREGROUND_CLASSES
    ).astype(np.uint8)

    # Resize mask to original image dimensions.
    mask_image = Image.fromarray(mask * 255)

    mask_image = mask_image.resize(
        original_size,
        Image.Resampling.NEAREST
    )

    mask = (
        np.asarray(mask_image) > 0
    ).astype(np.uint8)

    # Expand mask to protect hair and foreground boundaries.
    kernel = np.ones(
        (DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE),
        dtype=np.uint8
    )

    protected_mask = cv2.dilate(
        mask,
        kernel,
        iterations=1
    )

    return mask, protected_mask


# =====================================================
# 5. REPLACE BACKGROUND WITHOUT BLENDING
# =====================================================

def replace_background(original, protected_mask, color):

    # Start with a copy of the original.
    output = original.copy()

    # Replace only pixels outside the protected foreground.
    output[protected_mask == 0] = color

    return output


# =====================================================
# 6. INITIALIZE MEDIAPIPE
# =====================================================

options = mp.tasks.vision.ImageSegmenterOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    output_category_mask=True,
    output_confidence_masks=False,
)


# =====================================================
# 7. PROCESS IMAGES
# =====================================================

records = []

print("\nStarting controlled perturbation...\n")

with mp.tasks.vision.ImageSegmenter.create_from_options(
    options
) as segmenter:

    for position, row in samples.iterrows():

        image_index = int(row["image_index"])
        label = str(row["label"])

        print(
            f"[{position + 1}/60] "
            f"Image {image_index} - {label}"
        )

        original_image = (
            train_dataset[image_index]["image"]
            .convert("RGB")
        )

        original = np.asarray(original_image)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=original
        )

        result = segmenter.segment(mp_image)

        raw_mask, protected_mask = (
            create_preservation_mask(
                result,
                original_image.size
            )
        )

        foreground_pixels = (
            protected_mask.astype(bool)
        )

        foreground_area = (
            foreground_pixels.mean() * 100
        )

        record = {
            "image_index": image_index,
            "label": label,
            "raw_foreground_area_pct": float(
                raw_mask.mean() * 100
            ),
            "protected_foreground_area_pct": float(
                foreground_area
            ),
        }

        # Verify mask dimensions.
        if protected_mask.shape != original.shape[:2]:
            raise ValueError(
                f"Mask shape mismatch for image {image_index}"
            )

        for background_name, color in BACKGROUNDS.items():

            composite = replace_background(
                original,
                protected_mask,
                color
            )

            output_dir = OUTPUT_DIR / background_name
            output_dir.mkdir(parents=True, exist_ok=True)

            output_path = (
                output_dir / f"image_{image_index}.png"
            )

            Image.fromarray(composite).save(output_path)

            # Verify exact preservation of protected pixels.
            difference = np.abs(
                composite.astype(np.int16)
                - original.astype(np.int16)
            )

            if np.any(difference[foreground_pixels] != 0):
                raise AssertionError(
                    f"Foreground changed in image {image_index}"
                )

            changed_background_pixels = (
                np.any(difference != 0, axis=2)
                & (~foreground_pixels)
            ).sum()

            record[
                f"{background_name}_changed_pixels"
            ] = int(changed_background_pixels)

            record[
                f"{background_name}_foreground_preserved"
            ] = True

        records.append(record)


# =====================================================
# 8. SAVE MANIFEST
# =====================================================

manifest = pd.DataFrame(records)

manifest_path = OUTPUT_DIR / "controlled_manifest.csv"

manifest.to_csv(
    manifest_path,
    index=False
)


# =====================================================
# 9. SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("CONTROLLED BACKGROUND PERTURBATION COMPLETED")
print("=" * 60)

print("Images processed:", len(records))

print(
    "Mean raw foreground area:",
    round(manifest["raw_foreground_area_pct"].mean(), 2),
    "%"
)

print(
    "Mean protected foreground area:",
    round(manifest["protected_foreground_area_pct"].mean(), 2),
    "%"
)

print(
    "Foreground preservation:",
    all(
        manifest["gray_foreground_preserved"]
        & manifest["white_foreground_preserved"]
    )
)

print("Manifest:", manifest_path)
print("Output:", OUTPUT_DIR)