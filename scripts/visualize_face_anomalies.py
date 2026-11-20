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
    / "anomaly_face_detection_comparison.png"
)

DATASET_ID = "menes000/armocromia-12-season"

CASES = [
    ("train", 986),
    ("train", 1112),
    ("test", 404),
    ("test", 710),
    ("test", 895),
    ("test", 899),
]

THRESHOLDS = [0.50, 0.65]

PANEL_WIDTH = 340
PANEL_HEIGHT = 300

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def create_detector(threshold):
    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=threshold,
    )

    return vision.FaceDetector.create_from_options(options)


def annotate_image(image, detections, threshold):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for number, detection in enumerate(detections, start=1):
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

    preview = annotated.resize((224, 224))

    panel = Image.new(
        "RGB",
        (PANEL_WIDTH, PANEL_HEIGHT),
        "white",
    )

    panel_draw = ImageDraw.Draw(panel)

    panel_draw.text(
        (10, 8),
        f"Threshold: {threshold:.2f}",
        fill="black",
    )

    panel_draw.text(
        (10, 27),
        f"Detections: {len(detections)}",
        fill="black",
    )

    panel.paste(preview, (58, 60))

    return panel


print("Loading dataset...")
dataset = load_dataset(DATASET_ID)

detectors = {
    threshold: create_detector(threshold)
    for threshold in THRESHOLDS
}

cards = []

try:
    for split, index in CASES:
        row = dataset[split][index]

        image = row["image"].convert("RGB")
        array = np.ascontiguousarray(np.asarray(image))

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=array,
        )

        label = dataset[split].features["label"].int2str(
            row["label"]
        )

        panels = []

        for threshold in THRESHOLDS:
            result = detectors[threshold].detect(mp_image)
            detections = result.detections

            panels.append(
                annotate_image(image, detections, threshold)
            )

            print(
                f"{split} index {index} | "
                f"{label} | threshold={threshold:.2f} | "
                f"detections={len(detections)}"
            )

        card = Image.new(
            "RGB",
            (PANEL_WIDTH * 2, PANEL_HEIGHT + 45),
            "#dddddd",
        )

        draw = ImageDraw.Draw(card)

        draw.text(
            (10, 8),
            f"{split.upper()} | Index {index} | Label: {label}",
            fill="black",
        )

        card.paste(panels[0], (0, 45))
        card.paste(panels[1], (PANEL_WIDTH, 45))

        cards.append(card)

finally:
    for detector in detectors.values():
        detector.close()


columns = 2
rows = (len(cards) + columns - 1) // columns

sheet = Image.new(
    "RGB",
    (
        columns * PANEL_WIDTH * 2,
        rows * (PANEL_HEIGHT + 45),
    ),
    "white",
)

for i, card in enumerate(cards):
    x = (i % columns) * PANEL_WIDTH * 2
    y = (i // columns) * (PANEL_HEIGHT + 45)

    sheet.paste(card, (x, y))


sheet.save(OUTPUT_PATH)

print("\n" + "=" * 60)
print("FACE DETECTION COMPARISON COMPLETED")
print("Output:", OUTPUT_PATH)
print("=" * 60)