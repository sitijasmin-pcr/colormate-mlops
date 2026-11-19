from datasets import load_dataset
from PIL import Image, ImageDraw
from pathlib import Path
import csv

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_NAME = "menes000/armocromia-12-season"

REPORT_PATH = Path(
    "reports/dataset_audit/duplicate_report.csv"
)

OUTPUT_DIR = Path("reports/dataset_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_DISTANCE = 2
MAX_PAIRS = 25

IMAGE_WIDTH = 220
IMAGE_HEIGHT = 220
HEADER_HEIGHT = 65

# ==========================================
# LOAD DUPLICATE REPORT
# ==========================================

print("Reading duplicate report...")

with open(REPORT_PATH, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    candidates = [
        row for row in reader
        if row["duplicate_type"] == "near_duplicate_candidate"
        and row["split_a"] != row["split_b"]
        and int(row["distance"]) <= MAX_DISTANCE
    ]

candidates.sort(key=lambda row: int(row["distance"]))
candidates = candidates[:MAX_PAIRS]

print(f"Selected pairs: {len(candidates)}")

# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...")
dataset = load_dataset(DATASET_NAME)

# ==========================================
# CREATE CONTACT SHEET
# ==========================================

pair_height = IMAGE_HEIGHT + HEADER_HEIGHT

sheet = Image.new(
    "RGB",
    (
        IMAGE_WIDTH * 2,
        pair_height * len(candidates)
    ),
    "white"
)

draw = ImageDraw.Draw(sheet)

for pair_number, pair in enumerate(candidates):

    split_a = pair["split_a"]
    split_b = pair["split_b"]

    index_a = int(pair["index_a"])
    index_b = int(pair["index_b"])

    image_a = dataset[split_a][index_a]["image"].convert("RGB")
    image_b = dataset[split_b][index_b]["image"].convert("RGB")

    image_a.thumbnail((IMAGE_WIDTH - 10, IMAGE_HEIGHT - 10))
    image_b.thumbnail((IMAGE_WIDTH - 10, IMAGE_HEIGHT - 10))

    y = pair_number * pair_height

    draw.text(
        (5, y + 5),
        f"PAIR {pair_number + 1} | Distance: {pair['distance']}",
        fill="black"
    )

    draw.text(
        (5, y + 25),
        f"{split_a} #{index_a} | {pair['label_a']}",
        fill="black"
    )

    draw.text(
        (IMAGE_WIDTH + 5, y + 25),
        f"{split_b} #{index_b} | {pair['label_b']}",
        fill="black"
    )

    x_a = (IMAGE_WIDTH - image_a.width) // 2
    x_b = IMAGE_WIDTH + (IMAGE_WIDTH - image_b.width) // 2

    image_y_a = y + HEADER_HEIGHT
    image_y_b = y + HEADER_HEIGHT

    sheet.paste(image_a, (x_a, image_y_a))
    sheet.paste(image_b, (x_b, image_y_b))

    draw.line(
        (IMAGE_WIDTH, y, IMAGE_WIDTH, y + pair_height),
        fill="gray",
        width=2
    )

# ==========================================
# SAVE
# ==========================================

output_path = OUTPUT_DIR / "duplicate_pairs_review.png"
sheet.save(output_path)

print("\nVISUALIZATION COMPLETED")
print(f"Pairs visualized: {len(candidates)}")
print(f"Output: {output_path}")