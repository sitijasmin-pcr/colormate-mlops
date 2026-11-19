from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "cnn_sensitivity/background_sensitivity_predictions.csv"
)

PERTURBED_DIR = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "perturbed_images"
)

OUTPUT_DIR = (
    ROOT / "reports/dataset_audit/background_perturbation/"
    "cnn_sensitivity/visual_review"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    "original",
    "gray_t50",
    "white_t50",
    "gray_t70",
    "white_t70",
]

IMAGE_W = 190
IMAGE_H = 190
CAPTION_H = 65
GAP = 12
SAMPLES_PER_PAGE = 4

# --------------------------------------------------
# 1. Load prediction results
# --------------------------------------------------

print("Loading prediction results...")

predictions = pd.read_csv(PREDICTIONS_PATH)

required_columns = {
    "image_index",
    "true_label",
    "condition",
    "predicted_label",
    "confidence",
    "correct",
    "prediction_changed_from_original",
}

missing = required_columns - set(predictions.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# --------------------------------------------------
# 2. Select images changed in all four conditions
# --------------------------------------------------

changed = predictions[
    predictions["condition"].isin(CONDITIONS[1:])
].pivot(
    index="image_index",
    columns="condition",
    values="prediction_changed_from_original"
)

selected_indices = changed[
    changed[CONDITIONS[1:]].all(axis=1)
].index.tolist()

print(f"Selected images: {len(selected_indices)}")

if len(selected_indices) != 37:
    print(
        "WARNING: Selected count differs from expected 37. "
        "Check the prediction CSV."
    )

# --------------------------------------------------
# 3. Prepare prediction lookup
# --------------------------------------------------

prediction_lookup = predictions.set_index(
    ["image_index", "condition"]
)

# --------------------------------------------------
# 4. Load original dataset
# --------------------------------------------------

print("Loading original Hugging Face dataset...")

dataset = load_dataset("menes000/armocromia-12-season")

# --------------------------------------------------
# 5. Helper functions
# --------------------------------------------------

def load_variant(image_index, condition):

    if condition == "original":
        return dataset["train"][int(image_index)]["image"].convert("RGB")

    image_path = (
        PERTURBED_DIR
        / condition
        / f"image_{int(image_index)}.png"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return Image.open(image_path).convert("RGB")


def resize_image(image):

    image.thumbnail((IMAGE_W, IMAGE_H))

    canvas = Image.new(
        "RGB",
        (IMAGE_W, IMAGE_H),
        "white"
    )

    x = (IMAGE_W - image.width) // 2
    y = (IMAGE_H - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


def get_prediction(image_index, condition):

    row = prediction_lookup.loc[(image_index, condition)]

    return (
        str(row["predicted_label"]),
        float(row["confidence"]),
        bool(row["correct"])
    )


# --------------------------------------------------
# 6. Create contact sheets
# --------------------------------------------------

def create_contact_sheet(indices, page_number):

    columns = len(CONDITIONS)
    rows = len(indices)

    cell_width = IMAGE_W
    cell_height = IMAGE_H + CAPTION_H

    sheet_width = (
        GAP + columns * (cell_width + GAP)
    )

    sheet_height = (
        GAP + rows * (cell_height + GAP) + 35
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "#eeeeee"
    )

    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 13)
        title_font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    draw.text(
        (GAP, 5),
        f"High Background Sensitivity - Page {page_number}",
        fill="black",
        font=title_font
    )

    for row_number, image_index in enumerate(indices):

        true_label = str(
            predictions[
                predictions["image_index"] == image_index
            ]["true_label"].iloc[0]
        )

        for col_number, condition in enumerate(CONDITIONS):

            x = GAP + col_number * (cell_width + GAP)
            y = 35 + GAP + row_number * (cell_height + GAP)

            image = load_variant(image_index, condition)
            image = resize_image(image)

            sheet.paste(image, (x, y))

            predicted, confidence, correct = get_prediction(
                image_index,
                condition
            )

            status = "CORRECT" if correct else "WRONG"

            caption = [
                condition,
                f"Pred: {predicted}",
                f"Conf: {confidence:.1%} | {status}",
            ]

            draw.text(
                (x, y + IMAGE_H + 3),
                "\n".join(caption),
                fill="black",
                font=font,
                spacing=2
            )

        # Print true label above the row
        draw.text(
            (GAP, y - 15),
            f"Image {image_index} | True: {true_label}",
            fill="#174a78",
            font=font
        )

    output_path = (
        OUTPUT_DIR
        / f"high_sensitivity_page_{page_number}.png"
    )

    sheet.save(output_path)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# 7. Generate all pages
# --------------------------------------------------

for page_number, start in enumerate(
    range(0, len(selected_indices), SAMPLES_PER_PAGE),
    start=1
):

    page_indices = selected_indices[
        start:start + SAMPLES_PER_PAGE
    ]

    create_contact_sheet(page_indices, page_number)


# --------------------------------------------------
# 8. Save selected image list
# --------------------------------------------------

selected_df = predictions[
    predictions["image_index"].isin(selected_indices)
    & predictions["condition"].isin(CONDITIONS)
].copy()

selected_df.to_csv(
    OUTPUT_DIR / "high_sensitivity_predictions.csv",
    index=False
)

print("\n======================================")
print("VISUAL AUDIT COMPLETED")
print("======================================")
print(f"Images selected: {len(selected_indices)}")
print(f"Pages created: {(len(selected_indices) + SAMPLES_PER_PAGE - 1) // SAMPLES_PER_PAGE}")
print(f"Output directory: {OUTPUT_DIR}")