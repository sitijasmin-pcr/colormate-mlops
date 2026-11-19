from datasets import load_dataset
from PIL import Image, ImageDraw
from pathlib import Path
import pandas as pd

DATASET_NAME = "menes000/armocromia-12-season"

CSV_PATH = Path(
    "reports/dataset_audit/full_background_audit.csv"
)

OUTPUT_PATH = Path(
    "reports/dataset_audit/background_extremes.png"
)

SAMPLES_PER_GROUP = 3
IMAGE_SIZE = 160

print("Loading dataset and audit results...")

dataset = load_dataset(DATASET_NAME)
train_data = dataset["train"]

df = pd.read_csv(CSV_PATH)

class_names = train_data.features["label"].names

# 12 classes x 2 groups x 3 images
columns = 6
rows = len(class_names) * 2

cell_width = IMAGE_SIZE
cell_height = IMAGE_SIZE + 45

sheet = Image.new(
    "RGB",
    (columns * cell_width, rows * cell_height),
    "white"
)

draw = ImageDraw.Draw(sheet)

for class_id, class_name in enumerate(class_names):

    class_df = df[df["label"] == class_name]

    groups = [
        ("HIGHEST", class_df.nlargest(
            SAMPLES_PER_GROUP, "black_ratio"
        )),
        ("LOWEST", class_df.nsmallest(
            SAMPLES_PER_GROUP, "black_ratio"
        ))
    ]

    for group_index, (group_name, selected) in enumerate(groups):

        row = class_id * 2 + group_index

        for col, (_, record) in enumerate(selected.iterrows()):

            index = int(record["index"])

            image = train_data[index]["image"].convert("RGB")
            image.thumbnail((IMAGE_SIZE, IMAGE_SIZE))

            x = col * cell_width
            y = row * cell_height

            sheet.paste(image, (x, y))

            label_text = (
                f"{class_name} | {group_name}\n"
                f"Black: {record['black_ratio']:.1f}%"
            )

            draw.text(
                (x + 3, y + IMAGE_SIZE + 3),
                label_text,
                fill="black"
            )

            print(
                f"{class_name} {group_name}: "
                f"index={index}, "
                f"black={record['black_ratio']:.2f}%"
            )

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

sheet.save(OUTPUT_PATH)

print("\n" + "=" * 50)
print("BACKGROUND EXTREMES VISUALIZATION COMPLETED")
print("Output:", OUTPUT_PATH)
print("=" * 50)