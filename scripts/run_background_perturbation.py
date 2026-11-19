from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe as mp

from datasets import load_dataset
from PIL import Image


# =========================================================
# 1. PATH DAN KONFIGURASI
# =========================================================

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
    "perturbed_images"
)

SHEET_DIR = OUTPUT_DIR / "contact_sheets"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHEET_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.50, 0.70]

BACKGROUNDS = {
    "gray": np.array([128, 128, 128], dtype=np.float32),
    "white": np.array([255, 255, 255], dtype=np.float32),
}


# =========================================================
# 2. VALIDASI INPUT
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

if not SAMPLES_CSV.exists():
    raise FileNotFoundError(f"CSV sampel tidak ditemukan: {SAMPLES_CSV}")

samples = pd.read_csv(SAMPLES_CSV)

if "image_index" not in samples.columns:
    raise ValueError("Kolom image_index tidak ditemukan.")

if len(samples) != 60:
    raise ValueError(
        f"Diharapkan 60 sampel, ditemukan {len(samples)}."
    )

print("Jumlah sampel:", len(samples))


# =========================================================
# 3. LOAD DATASET
# =========================================================

print("\nMemuat dataset Hugging Face...")

dataset = load_dataset("menes000/armocromia-12-season")
train_dataset = dataset["train"]


# =========================================================
# 4. FUNGSI PERTURBASI
# =========================================================

def create_foreground_confidence(result):
    """
    Menggabungkan confidence kelas 1-5 sebagai
    confidence foreground.

    Kelas 0 adalah background.
    """

    if result.confidence_masks is None:
        raise RuntimeError("Confidence masks tidak tersedia.")

    if len(result.confidence_masks) != 6:
        raise RuntimeError(
            f"Seharusnya ada 6 confidence masks, "
            f"ditemukan {len(result.confidence_masks)}."
        )

    foreground_confidence = np.zeros(
        result.confidence_masks[0].numpy_view().squeeze().shape,
        dtype=np.float32
    )

    for class_id in range(1, 6):
        class_mask = (
            result.confidence_masks[class_id]
            .numpy_view()
            .squeeze()
        )

        foreground_confidence += class_mask.astype(np.float32)

    return np.clip(foreground_confidence, 0.0, 1.0)


def create_perturbed_image(
    original,
    foreground_confidence,
    threshold,
    background_color
):
    """
    Membuat komposit dengan alpha mask berbasis
    confidence foreground.

    Piksel dengan confidence rendah lebih transparan.
    Piksel dengan confidence tinggi lebih dipertahankan.
    """

    original_float = original.astype(np.float32)

    # Threshold menghasilkan alpha 0 sampai 1.
    alpha = np.clip(
        (foreground_confidence - threshold) / (1.0 - threshold),
        0.0,
        1.0
    )

    alpha = alpha[..., np.newaxis]

    background = np.broadcast_to(
        background_color,
        original_float.shape
    )

    composite = (
        original_float * alpha
        + background * (1.0 - alpha)
    )

    return np.clip(composite, 0, 255).astype(np.uint8), alpha.squeeze()


# =========================================================
# 5. INISIALISASI SEGMENTER
# =========================================================

options = mp.tasks.vision.ImageSegmenterOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    output_category_mask=True,
    output_confidence_masks=True,
)


# =========================================================
# 6. PROSES 60 GAMBAR
# =========================================================

records = []
visual_items = []

print("\nMemulai background perturbation...\n")

with mp.tasks.vision.ImageSegmenter.create_from_options(
    options
) as segmenter:

    for position, row in samples.iterrows():

        image_index = int(row["image_index"])
        label = str(row["label"])

        print(
            f"[{position + 1}/60] "
            f"Index {image_index} - {label}"
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

        foreground_confidence = create_foreground_confidence(
            result
        )

        # Pastikan resolusi confidence sama dengan gambar.
        if foreground_confidence.shape != original.shape[:2]:
            resized = Image.fromarray(
                foreground_confidence.astype(np.float32),
                mode="F"
            ).resize(
                original_image.size,
                Image.Resampling.BILINEAR
            )

            foreground_confidence = np.asarray(
                resized,
                dtype=np.float32
            )

        record = {
            "image_index": image_index,
            "label": label,
            "foreground_confidence_mean": float(
                foreground_confidence.mean()
            ),
        }

        item = {
            "image_index": image_index,
            "label": label,
            "original": original,
            "variants": {},
        }

        for threshold in THRESHOLDS:

            # Area yang melewati threshold.
            hard_mask = foreground_confidence >= threshold

            record[
                f"foreground_area_threshold_{threshold:.2f}_pct"
            ] = float(hard_mask.mean() * 100)

            for background_name, background_color in BACKGROUNDS.items():

                composite, alpha = create_perturbed_image(
                    original=original,
                    foreground_confidence=foreground_confidence,
                    threshold=threshold,
                    background_color=background_color
                )

                variant_name = (
                    f"{background_name}_t{int(threshold * 100)}"
                )

                image_dir = OUTPUT_DIR / variant_name
                image_dir.mkdir(parents=True, exist_ok=True)

                output_path = (
                    image_dir / f"image_{image_index}.png"
                )

                Image.fromarray(composite).save(output_path)

                item["variants"][variant_name] = composite

                record[
                    f"mean_alpha_{variant_name}"
                ] = float(alpha.mean())

        records.append(record)
        visual_items.append(item)


# =========================================================
# 7. SIMPAN CSV
# =========================================================

audit_df = pd.DataFrame(records)

csv_path = OUTPUT_DIR / "background_perturbation_manifest.csv"

audit_df.to_csv(csv_path, index=False)

print("\nManifest disimpan:", csv_path)


# =========================================================
# 8. BUAT CONTACT SHEETS
# =========================================================

variant_order = [
    "gray_t50",
    "white_t50",
    "gray_t70",
    "white_t70",
]

column_titles = [
    "Original",
    "Gray - T50",
    "White - T50",
    "Gray - T70",
    "White - T70",
]

print("\nMembuat contact sheets...")

for sheet_number in range(6):

    start = sheet_number * 10
    end = min(start + 10, len(visual_items))

    selected = visual_items[start:end]

    fig, axes = plt.subplots(
        len(selected),
        5,
        figsize=(15, 2.5 * len(selected)),
        squeeze=False
    )

    for row_number, item in enumerate(selected):

        axes[row_number, 0].imshow(item["original"])

        for col_number, variant_name in enumerate(
            variant_order,
            start=1
        ):
            axes[row_number, col_number].imshow(
                item["variants"][variant_name]
            )

        axes[row_number, 0].set_ylabel(
            f"{item['image_index']}\n{item['label']}",
            fontsize=7
        )

        for ax in axes[row_number]:
            ax.axis("off")

    for col_number, title in enumerate(column_titles):
        axes[0, col_number].set_title(
            title,
            fontsize=10,
            fontweight="bold"
        )

    fig.suptitle(
        f"ColorMate Background Perturbation "
        f"- Samples {start + 1} to {end}",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    sheet_path = (
        SHEET_DIR /
        f"perturbation_contact_sheet_{sheet_number + 1}.png"
    )

    plt.savefig(
        sheet_path,
        dpi=110,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Contact sheet:", sheet_path)


# =========================================================
# 9. RINGKASAN
# =========================================================

print("\n" + "=" * 55)
print("BACKGROUND PERTURBATION SELESAI")
print("=" * 55)

print("Jumlah gambar:", len(records))
print("Jumlah threshold:", len(THRESHOLDS))
print("Jumlah background:", len(BACKGROUNDS))
print("Jumlah variasi perturbasi:", len(records) * 4)
print("Manifest:", csv_path)
print("Output:", OUTPUT_DIR)