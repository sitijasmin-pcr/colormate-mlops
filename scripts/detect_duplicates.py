from datasets import load_dataset
from PIL import Image
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

# ==========================================
# CONFIGURATION
# ==========================================

DATASET_NAME = "menes000/armocromia-12-season"

OUTPUT_DIR = Path("reports/dataset_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Maximum Hamming distance for near-duplicate candidates.
# This is a heuristic threshold, not proof of duplication.
DHASH_THRESHOLD = 6

# ==========================================
# HASH FUNCTIONS
# ==========================================

def calculate_exact_hash(image):
    """
    Hash the decoded RGB pixels.
    Images with identical dimensions and pixel values
    will have the same hash.
    """
    image = image.convert("RGB")

    image_data = (
        str(image.size).encode("utf-8")
        + image.tobytes()
    )

    return hashlib.sha256(image_data).hexdigest()


def calculate_dhash(image):
    """
    Calculate a 64-bit Difference Hash (dHash).
    """
    image = image.convert("L").resize((9, 8))

    pixels = list(image.getdata())

    hash_value = 0

    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]

            hash_value = (hash_value << 1) | (
                1 if left > right else 0
            )

    return hash_value


def hamming_distance(hash_a, hash_b):
    """
    Count differing bits between two hashes.
    """
    return (hash_a ^ hash_b).bit_count()


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

records = []

# ==========================================
# CALCULATE IMAGE HASHES
# ==========================================

for split_name in ["train", "test"]:

    split_data = dataset[split_name]
    class_names = split_data.features["label"].names

    print(f"\nProcessing {split_name}...")

    for index in range(len(split_data)):

        sample = split_data[index]

        image = sample["image"].convert("RGB")

        label_id = sample["label"]
        label_name = class_names[label_id]

        records.append({
            "split": split_name,
            "index": index,
            "label": label_name,
            "exact_hash": calculate_exact_hash(image),
            "dhash": calculate_dhash(image),
        })

        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1} images")

print(f"\nTotal images processed: {len(records)}")

# ==========================================
# EXACT DUPLICATE DETECTION
# ==========================================

print("\nChecking exact duplicates...")

exact_groups = defaultdict(list)

for record in records:
    exact_groups[record["exact_hash"]].append(record)

exact_pairs = []

for group in exact_groups.values():

    if len(group) > 1:

        for i in range(len(group)):
            for j in range(i + 1, len(group)):

                a = group[i]
                b = group[j]

                exact_pairs.append({
                    "duplicate_type": "exact_pixel_duplicate",
                    "split_a": a["split"],
                    "index_a": a["index"],
                    "label_a": a["label"],
                    "split_b": b["split"],
                    "index_b": b["index"],
                    "label_b": b["label"],
                    "distance": 0,
                })

# ==========================================
# NEAR DUPLICATE DETECTION
# ==========================================

print("\nChecking near duplicates...")

near_pairs = []

for i in range(len(records)):

    for j in range(i + 1, len(records)):

        a = records[i]
        b = records[j]

        # Skip exact duplicates already reported.
        if a["exact_hash"] == b["exact_hash"]:
            continue

        distance = hamming_distance(
            a["dhash"],
            b["dhash"]
        )

        if distance <= DHASH_THRESHOLD:

            near_pairs.append({
                "duplicate_type": "near_duplicate_candidate",
                "split_a": a["split"],
                "index_a": a["index"],
                "label_a": a["label"],
                "split_b": b["split"],
                "index_b": b["index"],
                "label_b": b["label"],
                "distance": distance,
            })

# ==========================================
# SAVE CSV REPORT
# ==========================================

all_pairs = exact_pairs + near_pairs

report_path = OUTPUT_DIR / "duplicate_report.csv"

fieldnames = [
    "duplicate_type",
    "split_a",
    "index_a",
    "label_a",
    "split_b",
    "index_b",
    "label_b",
    "distance",
]

with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_pairs)

# ==========================================
# SUMMARY
# ==========================================

cross_split_exact = sum(
    1 for pair in exact_pairs
    if pair["split_a"] != pair["split_b"]
)

cross_split_near = sum(
    1 for pair in near_pairs
    if pair["split_a"] != pair["split_b"]
)

print("\n" + "=" * 50)
print("DUPLICATE DETECTION COMPLETED")
print("=" * 50)

print(f"Total images: {len(records)}")
print(f"Exact duplicate pairs: {len(exact_pairs)}")
print(f"Near duplicate candidates: {len(near_pairs)}")

print(f"Cross-split exact pairs: {cross_split_exact}")
print(f"Cross-split near candidates: {cross_split_near}")

print(f"\nReport saved to: {report_path}")