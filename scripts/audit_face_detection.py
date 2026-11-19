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

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_ID = "menes000/armocromia-12-season"

MODEL_PATH = (
    ROOT_DIR
    / "data"
    / "external"
    / "models"
    / "face_detection"
    / "blaze_face_short_range.tflite"
)

OUTPUT_DIR = ROOT_DIR / "reports" / "dataset_audit"

MIN_DETECTION_CONFIDENCE = 0.5


# ============================================================
# VALIDATION
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Face detection model not found: {MODEL_PATH}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("COLORMATE - FACE DETECTION DATASET AUDIT")
print("=" * 60)

print("\nLoading dataset...")

dataset = load_dataset(DATASET_ID)

print("Dataset loaded successfully.")

# ============================================================
# CREATE FACE DETECTOR
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
)

detector = vision.FaceDetector.create_from_options(options)


# ============================================================
# AUDIT EACH SPLIT
# ============================================================

all_results = []

try:
    for split_name in dataset.keys():

        split_dataset = dataset[split_name]

        print("\n" + "=" * 60)
        print(f"Processing split: {split_name}")
        print(f"Total images: {len(split_dataset)}")
        print("=" * 60)

        for index, row in enumerate(split_dataset):

            try:
                image = row["image"].convert("RGB")
                image_array = np.asarray(image)
                image_array = np.ascontiguousarray(image_array)

                height, width, _ = image_array.shape

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=image_array,
                )

                detection_result = detector.detect(mp_image)

                detections = detection_result.detections

                label = dataset[split_name].features[
                    "label"
                ].int2str(row["label"])

                result = {
                    "split": split_name,
                    "image_index": index,
                    "label": label,
                    "image_width": width,
                    "image_height": height,
                    "face_count": len(detections),
                    "max_confidence": None,
                    "bbox_x": None,
                    "bbox_y": None,
                    "bbox_width": None,
                    "bbox_height": None,
                    "bbox_area_ratio": None,
                    "center_x_ratio": None,
                    "center_y_ratio": None,
                    "status": "success",
                }

                if detections:

                    best_detection = max(
                        detections,
                        key=lambda detection: (
                            detection.categories[0].score
                        ),
                    )

                    confidence = (
                        best_detection.categories[0].score
                    )

                    bbox = best_detection.bounding_box

                    result.update({
                        "max_confidence": float(confidence),
                        "bbox_x": bbox.origin_x,
                        "bbox_y": bbox.origin_y,
                        "bbox_width": bbox.width,
                        "bbox_height": bbox.height,
                        "bbox_area_ratio": (
                            bbox.width * bbox.height
                        ) / (width * height),
                        "center_x_ratio": (
                            bbox.origin_x + bbox.width / 2
                        ) / width,
                        "center_y_ratio": (
                            bbox.origin_y + bbox.height / 2
                        ) / height,
                    })

                all_results.append(result)

            except Exception as error:

                all_results.append({
                    "split": split_name,
                    "image_index": index,
                    "label": None,
                    "image_width": None,
                    "image_height": None,
                    "face_count": None,
                    "max_confidence": None,
                    "bbox_x": None,
                    "bbox_y": None,
                    "bbox_width": None,
                    "bbox_height": None,
                    "bbox_area_ratio": None,
                    "center_x_ratio": None,
                    "center_y_ratio": None,
                    "status": f"error: {type(error).__name__}: {error}",
                })

            processed = index + 1

            if processed % 250 == 0 or processed == len(split_dataset):
                print(
                    f"{split_name}: "
                    f"{processed}/{len(split_dataset)} images processed"
                )

finally:
    detector.close()


# ============================================================
# SAVE AUDIT RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

detail_path = OUTPUT_DIR / "face_detection_audit.csv"

results_df.to_csv(
    detail_path,
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# SUMMARY BY SPLIT AND LABEL
# ============================================================

valid_results = results_df[
    results_df["status"] == "success"
].copy()

summary = (
    valid_results
    .groupby(["split", "label"], dropna=False)
    .agg(
        image_count=("image_index", "count"),
        no_face=("face_count", lambda x: (x == 0).sum()),
        single_face=("face_count", lambda x: (x == 1).sum()),
        multiple_faces=("face_count", lambda x: (x > 1).sum()),
        average_confidence=("max_confidence", "mean"),
        average_bbox_area_ratio=("bbox_area_ratio", "mean"),
    )
    .reset_index()
)

summary_path = OUTPUT_DIR / "face_detection_summary.csv"

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("FACE DETECTION AUDIT COMPLETED")
print("=" * 60)

print("\nResults by split:")

for split_name in results_df["split"].unique():

    split_results = results_df[
        results_df["split"] == split_name
    ]

    valid = split_results[
        split_results["status"] == "success"
    ]

    print(f"\nSplit: {split_name}")
    print(f"Total images: {len(split_results)}")
    print(f"Successfully processed: {len(valid)}")
    print(f"Processing errors: {len(split_results) - len(valid)}")

    if len(valid) > 0:
        print(
            "No face detected:",
            int((valid["face_count"] == 0).sum()),
        )
        print(
            "Single face detected:",
            int((valid["face_count"] == 1).sum()),
        )
        print(
            "Multiple faces detected:",
            int((valid["face_count"] > 1).sum()),
        )

print("\nDetailed CSV:")
print(detail_path)

print("\nSummary CSV:")
print(summary_path)

print("\nAudit finished.")