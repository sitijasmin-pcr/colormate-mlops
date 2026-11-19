from pathlib import Path

import mediapipe as mp
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT / "data" / "external" / "models"
    / "face_detection" / "blaze_face_short_range.tflite"
)

OUTPUT_PATH = (
    ROOT / "reports" / "dataset_audit"
    / "anomaly_face_detection_review.png"
)

# Six anomalous images identified from the CSV
CASES = [
    ("train", 986),
    ("train", 1112),
    ("test", 404),
    ("test", 710),
    ("test", 895),
    ("test", 899),
]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
dataset = load_dataset("menes000/armocromia-12-season")

options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence=0.5,
)

detector = vision.FaceDetector.create_from_options(options)

cards = []

try:
    for split, index in CASES:
        row = dataset[split][index]

        image = row["image"].convert("RGB")
        array = np.ascontiguousarray(np.asarray(image))

        result = detector.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=array,
            )
        )

        label = dataset[split].features["label"].int2str(
            row["label"]
        )

        # Draw every detected bounding box
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for number, detection in enumerate(
            result.detections, start=1
        ):
            bbox = detection.bounding_box
            score = detection.categories[0].score

            x1 = bbox.origin_x
            y1 = bbox.origin_y
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height

            draw.rectangle(
                (x1, y1, x2, y2),
                outline="red",
                width=2,
            )

            draw.text(
                (x1, max(0, y1 - 14)),
                f"Face {number}: {score:.3f}",
                fill="red",
                stroke_width=1,
                stroke_fill="white",
            )

        # Create a card for each image
        card = Image.new("RGB", (340, 285), "white")
        card_draw = ImageDraw.Draw(card)

        card_draw.text(
            (10, 8),
            f"{split.upper()} | Index {index}",
            fill="black",
        )

        card_draw.text(
            (10, 25),
            f"Label: {label}",
            fill="black",
        )

        card_draw.text(
            (10, 42),
            f"Detections: {len(result.detections)}",
            fill="black",
        )

        preview = annotated.resize((224, 224))
        card.paste(preview, (58, 58))

        cards.append(card)

        print(
            f"{split} index {index} | "
            f"{label} | "
            f"{len(result.detections)} detection(s)"
        )

finally:
    detector.close()


# Arrange six cards in 3 columns x 2 rows
sheet = Image.new("RGB", (340 * 3, 285 * 2), "#dddddd")

for i, card in enumerate(cards):
    x = (i % 3) * 340
    y = (i // 3) * 285
    sheet.paste(card, (x, y))

sheet.save(OUTPUT_PATH)

print("\nVisualization saved:")
print(OUTPUT_PATH)