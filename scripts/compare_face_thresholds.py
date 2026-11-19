from pathlib import Path

import mediapipe as mp
import numpy as np
import pandas as pd
from datasets import load_dataset
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "data" / "external" / "models"
    / "face_detection" / "blaze_face_short_range.tflite"
)

CASES = [
    ("train", 986),
    ("train", 1112),
    ("test", 404),
    ("test", 710),
    ("test", 895),
    ("test", 899),
]

THRESHOLDS = [0.50, 0.65]

print("Loading dataset...")
dataset = load_dataset("menes000/armocromia-12-season")


def create_detector(threshold):
    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=threshold,
    )

    return vision.FaceDetector.create_from_options(options)


results = []

for threshold in THRESHOLDS:

    print("\n" + "=" * 65)
    print(f"TESTING CONFIDENCE THRESHOLD: {threshold}")
    print("=" * 65)

    detector = create_detector(threshold)

    try:
        for split, index in CASES:

            row = dataset[split][index]

            image = row["image"].convert("RGB")
            image_array = np.ascontiguousarray(
                np.asarray(image)
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image_array,
            )

            detection_result = detector.detect(mp_image)

            detections = detection_result.detections

            scores = sorted(
                [
                    round(
                        detection.categories[0].score,
                        4
                    )
                    for detection in detections
                ],
                reverse=True,
            )

            label = dataset[split].features[
                "label"
            ].int2str(row["label"])

            results.append({
                "threshold": threshold,
                "split": split,
                "image_index": index,
                "label": label,
                "detection_count": len(detections),
                "confidence_scores": str(scores),
            })

            print(
                f"{split:5} | index {index:4} | "
                f"{label:14} | "
                f"faces: {len(detections)} | "
                f"scores: {scores}"
            )

    finally:
        detector.close()


# Save comparison results
output_dir = ROOT / "reports" / "dataset_audit"
output_dir.mkdir(parents=True, exist_ok=True)

df = pd.DataFrame(results)

output_path = (
    output_dir / "face_threshold_comparison.csv"
)

df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 65)
print("THRESHOLD COMPARISON COMPLETED")
print("=" * 65)
print(f"Saved: {output_path}")