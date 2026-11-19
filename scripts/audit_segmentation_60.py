from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from datasets import load_dataset
from PIL import Image
import mediapipe as mp


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
    "segmentation_audit"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "background",
    "hair",
    "body_skin",
    "face_skin",
    "clothes",
    "others",
]

# Warna visual untuk setiap kelas
CLASS_COLORS = [
    "#222222",  # Background
    "#E69F00",  # Hair
    "#56B4E9",  # Body skin
    "#009E73",  # Face skin
    "#CC79A7",  # Clothes
    "#F0E442",  # Others
]

CATEGORY_CMAP = ListedColormap(CLASS_COLORS)


# =========================================================
# 2. VALIDASI FILE
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")

if not SAMPLES_CSV.exists():
    raise FileNotFoundError(f"CSV sampel tidak ditemukan: {SAMPLES_CSV}")


# =========================================================
# 3. LOAD DATASET DAN SAMPEL
# =========================================================

print("Memuat daftar sampel...")
samples = pd.read_csv(SAMPLES_CSV)

if "image_index" not in samples.columns:
    raise ValueError(
        "Kolom image_index tidak ditemukan dalam CSV sampel."
    )

if len(samples) != 60:
    raise ValueError(
        f"Jumlah sampel seharusnya 60, ditemukan {len(samples)}."
    )

print("Memuat dataset Hugging Face...")
dataset = load_dataset("menes000/armocromia-12-season")

train_dataset = dataset["train"]

print("Jumlah sampel:", len(samples))
print("Jumlah gambar train:", len(train_dataset))


# =========================================================
# 4. INISIALISASI IMAGE SEGMENTER
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
# 5. PROSES SEGMENTASI
# =========================================================

records = []
visual_items = []

print("\nMemulai audit segmentasi 60 gambar...\n")

with mp.tasks.vision.ImageSegmenter.create_from_options(
    options
) as segmenter:

    for position, row in samples.iterrows():

        image_index = int(row["image_index"])
        label = str(row["label"])

        print(
            f"[{position + 1}/60] "
            f"Memproses index {image_index} - {label}"
        )

        original = (
            train_dataset[image_index]["image"]
            .convert("RGB")
        )

        original_array = np.asarray(original)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=original_array
        )

        result = segmenter.segment(mp_image)

        if result.category_mask is None:
            raise RuntimeError(
                f"Category mask kosong pada index {image_index}"
            )

        category = result.category_mask.numpy_view().squeeze()

        # Pastikan ukuran mask sama dengan gambar asli
        if category.shape != original_array.shape[:2]:
            category_image = Image.fromarray(
                category.astype(np.uint8)
            )
            category = np.asarray(
                category_image.resize(
                    original.size,
                    Image.Resampling.NEAREST
                )
            )

        # Validasi nilai kategori
        if not np.isin(category, range(6)).all():
            raise ValueError(
                f"Kategori tidak valid pada index {image_index}"
            )

        total_pixels = category.size

        # Mask foreground: semua kelas selain background
        foreground_mask = category != 0

        # Background diganti abu-abu hanya untuk visualisasi
        foreground_preview = np.full_like(
            original_array,
            fill_value=225
        )

        foreground_preview[foreground_mask] = (
            original_array[foreground_mask]
        )

        # Hitung persentase setiap kategori
        counts = np.bincount(
            category.flatten(),
            minlength=6
        )

        percentages = (
            counts / total_pixels * 100
        )

        record = {
            "image_index": image_index,
            "label": label,
            "total_pixels": total_pixels,
            "foreground_ratio_pct": float(
                foreground_mask.mean() * 100
            ),
        }

        for class_id, class_name in enumerate(CLASS_NAMES):
            record[f"{class_name}_pct"] = float(
                percentages[class_id]
            )

        records.append(record)

        visual_items.append({
            "index": image_index,
            "label": label,
            "original": original_array,
            "category": category.copy(),
            "foreground": foreground_preview,
        })


# =========================================================
# 6. SIMPAN HASIL STATISTIK CSV
# =========================================================

audit_df = pd.DataFrame(records)

csv_output = OUTPUT_DIR / "segmentation_audit_60.csv"

audit_df.to_csv(csv_output, index=False)

print("\nCSV berhasil disimpan:")
print(csv_output)


# =========================================================
# 7. BUAT RINGKASAN KATEGORI
# =========================================================

summary = audit_df[
    [f"{name}_pct" for name in CLASS_NAMES]
].agg(["mean", "std", "min", "max"])

summary_output = OUTPUT_DIR / "segmentation_category_summary.csv"

summary.to_csv(summary_output)

print("\nRata-rata persentase kategori:")
print(summary.loc["mean"].round(2))

print("\nRingkasan disimpan:")
print(summary_output)


# =========================================================
# 8. BUAT CONTACT SHEET
# =========================================================

print("\nMembuat contact sheets...")

for sheet_number in range(6):

    start = sheet_number * 10
    end = min(start + 10, len(visual_items))

    selected_items = visual_items[start:end]

    fig, axes = plt.subplots(
        len(selected_items),
        3,
        figsize=(12, 2.7 * len(selected_items)),
        squeeze=False
    )

    for row_number, item in enumerate(selected_items):

        axes[row_number, 0].imshow(item["original"])
        axes[row_number, 0].set_title(
            f"Original | Index {item['index']} | {item['label']}",
            fontsize=8
        )

        axes[row_number, 1].imshow(
            item["category"],
            cmap=CATEGORY_CMAP,
            vmin=0,
            vmax=5,
            interpolation="nearest"
        )
        axes[row_number, 1].set_title(
            "Category Mask",
            fontsize=8
        )

        axes[row_number, 2].imshow(item["foreground"])
        axes[row_number, 2].set_title(
            "Foreground Preview",
            fontsize=8
        )

        for ax in axes[row_number]:
            ax.axis("off")

    fig.suptitle(
        f"ColorMate Segmentation Audit "
        f"- Samples {start + 1} to {end}",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    image_output = (
        OUTPUT_DIR /
        f"segmentation_contact_sheet_{sheet_number + 1}.png"
    )

    plt.savefig(
        image_output,
        dpi=110,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Contact sheet:", image_output)


# =========================================================
# 9. SELESAI
# =========================================================

print("\n" + "=" * 55)
print("AUDIT SEGMENTASI SELESAI")
print("=" * 55)

print("Total gambar diproses:", len(records))
print("Total kategori:", len(CLASS_NAMES))
print("Output folder:", OUTPUT_DIR)