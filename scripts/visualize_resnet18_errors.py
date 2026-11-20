from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image, ImageDraw
import math

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT / "reports" / "experiments"
    / "resnet18_experiment01" / "evaluation"
    / "validation_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "reports" / "experiments"
    / "resnet18_experiment01" / "evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "menes000/armocromia-12-season"

df = pd.read_csv(PREDICTIONS_PATH)

wrong = (
    df[df["correct"] == False]
    .sort_values("confidence", ascending=False)
    .head(20)
    .reset_index(drop=True)
)

print("Loading dataset...")
dataset = load_dataset(DATASET_ID)

source = dataset["train"]

columns = 4
rows = math.ceil(len(wrong) / columns)

cell_width = 320
cell_height = 370

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows * cell_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

for position, row in wrong.iterrows():

    index = int(row["image_index"])

    image = source[index]["image"].convert("RGB")
    image.thumbnail((280, 270))

    x = (position % columns) * cell_width
    y = (position // columns) * cell_height

    image_x = x + (cell_width - image.width) // 2
    image_y = y + 10

    sheet.paste(image, (image_x, image_y))

    text_y = y + 290

    lines = [
        f"Image index: {index}",
        f"True: {row['true_label']}",
        f"Pred: {row['predicted_label']}",
        f"Confidence: {row['confidence']:.2%}",
    ]

    for line in lines:
        draw.text(
            (x + 12, text_y),
            line,
            fill="black"
        )
        text_y += 19

    print(f"Processed {position + 1}/20: {index}")

output_path = OUTPUT_DIR / "top20_high_confidence_errors.png"

sheet.save(output_path)

print("\nContact sheet saved:")
print(output_path)
