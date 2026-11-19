from datasets import load_dataset
from PIL import Image, ImageDraw
from pathlib import Path
import random

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_NAME = "menes000/armocromia-12-season"

OUTPUT_DIR = Path("reports/dataset_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLASS = 10
RANDOM_SEED = 42

IMAGE_WIDTH = 180
IMAGE_HEIGHT = 180
LABEL_HEIGHT = 35
COLUMNS = 5

# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

train_data = dataset["train"]
class_names = train_data.features["label"].names

random_generator = random.Random(RANDOM_SEED)

# ==========================================
# COLLECT RANDOM SAMPLES
# ==========================================

samples_by_class = {
    class_id: []
    for class_id in range(len(class_names))
}

print("\nSelecting random samples...")

for class_id, class_name in enumerate(class_names):

    class_indices = [
        index
        for index, label in enumerate(train_data["label"])
        if label == class_id
    ]

    selected_indices = random_generator.sample(
        class_indices,
        min(SAMPLES_PER_CLASS, len(class_indices))
    )

    samples_by_class[class_id] = selected_indices

    print(
        f"{class_name}: "
        f"{len(selected_indices)} random images"
    )

# ==========================================
# CREATE CONTACT SHEET
# ==========================================

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

for class_id, class_name in enumerate(class_names):

    for sample_number, sample_index in enumerate(
        samples_by_class[class_id]
    ):

        image = train_data[sample_index]["image"].convert("RGB")

        image.thumbnail(
            (IMAGE_WIDTH - 10, IMAGE_HEIGHT - 10)
        )

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

# ==========================================
# SAVE RESULTS
# ==========================================

output_path = (
    OUTPUT_DIR / "armocromia_random_sampling_120.png"
)

sheet.save(output_path)

print("\n" + "=" * 45)
print("RANDOM VISUAL SAMPLING COMPLETED")
print("=" * 45)

print(
    "Total samples:",
    sum(len(samples) for samples in samples_by_class.values())
)

print("Random seed:", RANDOM_SEED)
print("Output:", output_path)