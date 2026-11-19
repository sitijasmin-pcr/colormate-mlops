import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_PATH = Path(
    "reports/dataset_audit/face_region_audit.csv"
)

BACKGROUND_PATH = Path(
    "reports/dataset_audit/full_background_audit.csv"
)

OUTPUT_DIR = Path("reports/dataset_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading audit results...")

region_df = pd.read_csv(INPUT_PATH)
background_df = pd.read_csv(BACKGROUND_PATH)

df = region_df.merge(
    background_df[["index", "black_ratio"]],
    on="index",
    how="inner"
)

# ==========================================
# 1. BOUNDING BOX DISTRIBUTION
# ==========================================

plt.figure(figsize=(14, 7))

df.boxplot(
    column="bbox_area_ratio",
    by="label",
    rot=45
)

plt.title("Non-Black Bounding Box Distribution by Class")
plt.suptitle("")
plt.xlabel("Season Class")
plt.ylabel("Bounding Box Area (%)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "bbox_distribution_by_class.png",
    dpi=200
)

plt.close()


# ==========================================
# 2. BLACK PIXEL DISTRIBUTION
# ==========================================

plt.figure(figsize=(14, 7))

df.boxplot(
    column="black_ratio",
    by="label",
    rot=45
)

plt.title("Near-Black Pixel Distribution by Class")
plt.suptitle("")
plt.xlabel("Season Class")
plt.ylabel("Near-Black Pixel Ratio (%)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "black_pixel_distribution_by_class.png",
    dpi=200
)

plt.close()


# ==========================================
# 3. RELATIONSHIP BETWEEN BOTH METRICS
# ==========================================

plt.figure(figsize=(9, 7))

plt.scatter(
    df["black_ratio"],
    df["bbox_area_ratio"],
    alpha=0.35
)

plt.title(
    "Near-Black Pixels vs Non-Black Bounding Box"
)

plt.xlabel("Near-Black Pixel Ratio (%)")
plt.ylabel("Non-Black Bounding Box Area (%)")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "background_bbox_relationship.png",
    dpi=200
)

plt.close()


print("\nVisualizations saved:")
print("- bbox_distribution_by_class.png")
print("- black_pixel_distribution_by_class.png")
print("- background_bbox_relationship.png")

print("\nVISUALIZATION COMPLETED")