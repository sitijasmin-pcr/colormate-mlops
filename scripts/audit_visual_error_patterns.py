from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# CONFIGURATION
# =====================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

AUDIT_DIR = ROOT_DIR / "reports" / "dataset_audit"

PREDICTION_PATH = AUDIT_DIR / "baseline_validation_predictions.csv"
BACKGROUND_PATH = AUDIT_DIR / "full_background_audit.csv"
REGION_PATH = AUDIT_DIR / "face_region_audit.csv"

OUTPUT_DIR = AUDIT_DIR / "visual_error_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# LOAD DATA
# =====================================================

print("=" * 65)
print("COLORMATE - VISUAL ERROR PATTERN AUDIT")
print("=" * 65)

predictions = pd.read_csv(PREDICTION_PATH)
background = pd.read_csv(BACKGROUND_PATH)
region = pd.read_csv(REGION_PATH)

print("\nLoaded files:")
print("Predictions:", len(predictions))
print("Background audit:", len(background))
print("Region audit:", len(region))


# =====================================================
# VALIDATE AUDIT INDICES
# =====================================================

for name, dataframe in [
    ("background", background),
    ("region", region),
]:
    if dataframe["index"].duplicated().any():
        raise ValueError(f"Duplicate index found in {name} audit.")

    if dataframe["index"].isna().any():
        raise ValueError(f"Missing index found in {name} audit.")


# =====================================================
# MERGE DATA
# =====================================================

background_features = background.rename(
    columns={"index": "image_index"}
)

region_features = region.rename(
    columns={"index": "image_index"}
)

merged = predictions.merge(
    background_features,
    on="image_index",
    how="left",
    validate="one_to_one",
    suffixes=("", "_background")
)

merged = merged.merge(
    region_features,
    on="image_index",
    how="left",
    validate="one_to_one",
    suffixes=("", "_region")
)


# =====================================================
# DATA QUALITY CHECKS
# =====================================================

feature_columns = [
    "black_ratio",
    "border_black_ratio",
    "bbox_area_ratio",
    "non_black_ratio"
]

if merged[feature_columns].isna().any().any():
    missing = merged[feature_columns].isna().sum()
    raise ValueError(
        "Missing visual audit features:\n"
        + missing.to_string()
    )

if len(merged) != len(predictions):
    raise ValueError("Merged row count differs from predictions.")

if not (merged["label"] == merged["label_background"]).all():
    raise ValueError("Background audit labels do not match predictions.")

if not (merged["label"] == merged["label_region"]).all():
    raise ValueError("Region audit labels do not match predictions.")

print("\nData validation passed.")
print("Merged validation images:", len(merged))


# =====================================================
# OVERALL SUMMARY
# =====================================================

feature_summary = (
    merged.groupby("correct")[feature_columns]
    .agg(["mean", "median", "std", "min", "max"])
)

print("\nVISUAL FEATURES: CORRECT VS INCORRECT")
print(feature_summary.to_string())

feature_summary.to_csv(
    OUTPUT_DIR / "visual_features_correct_vs_incorrect.csv"
)


# =====================================================
# PER-CLASS SUMMARY
# =====================================================

class_summary = (
    merged.groupby(["label", "correct"])[feature_columns]
    .mean()
    .reset_index()
)

class_summary.to_csv(
    OUTPUT_DIR / "visual_features_by_class_and_correctness.csv",
    index=False
)

print("\nPER-CLASS VISUAL FEATURE SUMMARY")
print(class_summary.to_string(index=False))


# =====================================================
# SAVE MERGED DATA
# =====================================================

merged_path = OUTPUT_DIR / "validation_visual_error_analysis.csv"

merged.to_csv(merged_path, index=False)


# =====================================================
# BOXPLOT COMPARISON
# =====================================================

feature_titles = {
    "black_ratio": "Near-black Pixel Ratio",
    "border_black_ratio": "Border Near-black Ratio",
    "bbox_area_ratio": "Non-black Bounding Box Area Ratio",
    "non_black_ratio": "Non-black Pixel Ratio"
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, feature in zip(axes.flatten(), feature_columns):

    correct_values = merged.loc[
        merged["correct"] == True, feature
    ]

    incorrect_values = merged.loc[
        merged["correct"] == False, feature
    ]

    ax.boxplot(
        [correct_values, incorrect_values],
        tick_labels=["Correct", "Incorrect"]
    )

    ax.set_title(feature_titles[feature])
    ax.set_ylabel("Percentage (%)")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    "ColorMate CNN - Visual Features vs Prediction Correctness",
    fontsize=14
)

fig.tight_layout()

plot_path = OUTPUT_DIR / "visual_features_correct_vs_incorrect.png"

fig.savefig(
    plot_path,
    dpi=160,
    bbox_inches="tight"
)

plt.close(fig)


# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n" + "=" * 65)
print("VISUAL ERROR AUDIT COMPLETED")
print("=" * 65)

print("Correct predictions:", int(merged["correct"].sum()))
print("Incorrect predictions:", int((~merged["correct"]).sum()))

print("\nGenerated files:")
print(merged_path)
print(OUTPUT_DIR / "visual_features_correct_vs_incorrect.csv")
print(OUTPUT_DIR / "visual_features_by_class_and_correctness.csv")
print(plot_path)

print("\nAudit completed successfully.")