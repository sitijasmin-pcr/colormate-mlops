from datasets import load_dataset
from PIL import Image, ImageDraw
from pathlib import Path

DATASET_NAME = "menes000/armocromia-12-season"

OUTPUT_DIR = Path("reports/dataset_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLASS = 10
IMAGE_WIDTH = 180
IMAGE_HEIGHT = 180
LABEL_HEIGHT = 35
COLUMNS = 5

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

train_data = dataset["train"]
class_names = train_data.features["label"].names

ROWS = len(class_names) * 2

sheet = Image.new(
    "RGB",
    (
        COLUMNS * IMAGE_WIDTH,
        ROWS * (IMAGE_HEIGHT + LABEL_HEIGHT)
    ),
    "white"
)

draw = ImageDraw.Draw(sheet)

# Collect 10 sample indices for each class
samples_by_class = {i: [] for i in range(len(class_names))}

for index, label in enumerate(train_data["label"]):
    if len(samples_by_class[label]) < SAMPLES_PER_CLASS:
        samples_by_class[label].append(index)

# Draw images
for class_id, class_name in enumerate(class_names):
    sample_indices = samples_by_class[class_id]

    print(f"Processing {class_name}: {len(sample_indices)} images")

    for sample_number, sample_index in enumerate(sample_indices):
        image = train_data[sample_index]["image"].convert("RGB")
        image.thumbnail((IMAGE_WIDTH - 10, IMAGE_HEIGHT - 10))

        row = class_id * 2 + sample_number // COLUMNS
        col = sample_number % COLUMNS

        x = col * IMAGE_WIDTH
        y = row * (IMAGE_HEIGHT + LABEL_HEIGHT)

        image_x = x + (IMAGE_WIDTH - image.width) // 2
        image_y = y + (IMAGE_HEIGHT - image.height) // 2

        sheet.paste(image, (image_x, image_y))

        draw.text(
            (x + 5, y + IMAGE_HEIGHT + 5),
            f"{class_name} #{sample_number + 1}",
            fill="black"
        )

output_path = OUTPUT_DIR / "armocromia_contact_sheet_120.png"
sheet.save(output_path)

print("\nVISUALIZATION COMPLETED")
print(f"Total images: {sum(len(v) for v in samples_by_class.values())}")
print(f"Saved to: {output_path}")