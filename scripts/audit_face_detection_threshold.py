from pathlib import Path

import mediapipe as mp
import numpy as np
import pandas as pd
from datasets import load_dataset
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "menes000/armocromia-12-season"

MODEL_PATH = (
    ROOT / "data" / "external" / "models"
    / "face_detection" / "blaze_face_short_range.tflite"
)

OUTPUT_DIR = ROOT / "reports" / "dataset_audit"

BASELINE_PATH = OUTPUT_DIR / "face_detection_audit.csv"

THRESHOLD = 0.65

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not BASELINE_PATH.exists():
    raise FileNotFoundError(
        f"Baseline audit not found: {BASELINE_PATH}"
    )


# ============================================================
# LOAD BASELINE AND DATASET
# ============================================================

print("=" * 65)
print("COLORMATE - FACE DETECTION THRESHOLD AUDIT")
print("=" * 65)

baseline = pd.read_csv(BASELINE_PATH)

print("\nLoading dataset...")
dataset = load_dataset(DATASET_ID)

# ============================================================
# CREATE DETECTOR
# ============================================================

options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=THRESHOLD,
)

detector = vision.FaceDetector.create_from_options(options)

results = []

# ============================================================
# AUDIT ALL SPLITS
# ============================================================

try:
    for split_name in dataset.keys():

        split_dataset = dataset[split_name]

        print(f"\nProcessing {split_name}: {len(split_dataset)} images")

        for index, row in enumerate(split_dataset):

            try:
                image = row["image"].convert("RGB")
                image_array = np.ascontiguousarray(
                    np.asarray(image)
                )

                height, width, _ = image_array.shape

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=image_array,
                )

                detection_result = detector.detect(mp_image)
                detections = detection_result.detections

                label = split_dataset.features[
                    "label"
                ].int2str(row["label"])

                best = (
                    max(
                        detections,
                        key=lambda d: d.categories[0].score,
                    )
                    if detections
                    else None
                )

                record = {
                    "split": split_name,
                    "image_index": index,
                    "label": label,
                    "face_count": len(detections),
                    "max_confidence": None,
                    "bbox_x": None,
                    "bbox_y": None,
                    "bbox_width": None,
                    "bbox_height": None,
                    "bbox_area_ratio": None,
                    "status": "success",
                }

                if best is not None:
                    bbox = best.bounding_box

                    record.update({
                        "max_confidence": float(
                            best.categories[0].score
                        ),
                        "bbox_x": bbox.origin_x,
                        "bbox_y": bbox.origin_y,
                        "bbox_width": bbox.width,
                        "bbox_height": bbox.height,
                        "bbox_area_ratio": (
                            bbox.width * bbox.height
                        ) / (width * height),
                    })

                results.append(record)

            except Exception as error:
                results.append({
                    "split": split_name,
                    "image_index": index,
                    "label": None,
                    "face_count": None,
                    "max_confidence": None,
                    "bbox_x": None,
                    "bbox_y": None,
                    "bbox_width": None,
                    "bbox_height": None,
                    "bbox_area_ratio": None,
                    "status": (
                        f"error: {type(error).__name__}: {error}"
                    ),
                })

            processed = index + 1

            if processed % 250 == 0 or processed == len(split_dataset):
                print(
                    f"{split_name}: "
                    f"{processed}/{len(split_dataset)}"
                )

finally:
    detector.close()


# ============================================================
# SAVE THRESHOLD 0.65 RESULTS
# ============================================================

new_df = pd.DataFrame(results)

new_path = OUTPUT_DIR / "face_detection_audit_065.csv"

new_df.to_csv(
    new_path,
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# COMPARE WITH BASELINE 0.50
# ============================================================

keys = ["split", "image_index"]

comparison = baseline.merge(
    new_df,
    on=keys,
    how="outer",
    suffixes=("_050", "_065"),
    indicator=True,
)

comparison["face_count_change"] = (
    comparison["face_count_065"]
    - comparison["face_count_050"]
)

comparison_path = (
    OUTPUT_DIR / "face_detection_comparison_050_065.csv"
)

comparison.to_csv(
    comparison_path,
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# SUMMARY BY SPLIT AND LABEL
# ============================================================

def summarize(df, suffix):
    valid = df[df[f"status{suffix}"] == "success"].copy()

    return (
        valid.groupby(
            [f"split{suffix}", f"label{suffix}"],
            dropna=False,
        )
        .agg(
            images=(f"image_index", "count"),
            no_face=(f"face_count{suffix}", lambda x: (x == 0).sum()),
            single_face=(f"face_count{suffix}", lambda x: (x == 1).sum()),
            multiple_faces=(f"face_count{suffix}", lambda x: (x > 1).sum()),
            mean_confidence=(f"max_confidence{suffix}", "mean"),
        )
        .reset_index()
        .rename(columns={
            f"split{suffix}": "split",
            f"label{suffix}": "label",
        })
    )


summary_050 = summarize(comparison, "_050")
summary_065 = summarize(comparison, "_065")

summary = summary_050.merge(
    summary_065,
    on=["split", "label"],
    suffixes=("_050", "_065"),
)

for metric in [
    "no_face",
    "single_face",
    "multiple_faces",
]:
    summary[f"{metric}_change"] = (
        summary[f"{metric}_065"]
        - summary[f"{metric}_050"]
    )

summary_path = (
    OUTPUT_DIR / "face_detection_summary_050_065.csv"
)

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 65)
print("THRESHOLD COMPARISON SUMMARY")
print("=" * 65)

for split in new_df["split"].dropna().unique():

    part = new_df[
        (new_df["split"] == split)
        & (new_df["status"] == "success")
    ]

    print(f"\nSplit: {split}")
    print(f"Processed: {len(part)}")
    print(f"No face: {(part['face_count'] == 0).sum()}")
    print(f"Single face: {(part['face_count'] == 1).sum()}")
    print(f"Multiple faces: {(part['face_count'] > 1).sum()}")

print("\nProcessing errors:", (new_df["status"] != "success").sum())

print("\nComparison by split and label:")
print(
    summary[
        [
            "split",
            "label",
            "single_face_050",
            "single_face_065",
            "multiple_faces_050",
            "multiple_faces_065",
            "no_face_050",
            "no_face_065",
        ]
    ].to_string(index=False)
)

print("\nSaved files:")
print(new_path)
print(comparison_path)
print(summary_path)

print("\nThreshold audit completed.")