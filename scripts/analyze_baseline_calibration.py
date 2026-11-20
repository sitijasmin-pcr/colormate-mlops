from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIGURATION
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

PREDICTION_PATH = (
    ROOT_DIR
    / "reports"
    / "dataset_audit"
    / "baseline_validation_predictions.csv"
)

OUTPUT_DIR = ROOT_DIR / "reports" / "dataset_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_BINS = 10


# =========================================================
# LOAD PREDICTIONS
# =========================================================

print("=" * 65)
print("COLORMATE CNN BASELINE CALIBRATION ANALYSIS")
print("=" * 65)

if not PREDICTION_PATH.exists():
    raise FileNotFoundError(PREDICTION_PATH)

df = pd.read_csv(PREDICTION_PATH)

required_columns = [
    "label",
    "predicted_label",
    "confidence",
    "correct",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing required column: {column}")

probability_columns = [
    column for column in df.columns
    if column.startswith("prob_")
]

if len(df) != 760:
    raise ValueError(
        f"Expected 760 validation rows, got {len(df)}"
    )

if len(probability_columns) != 12:
    raise ValueError(
        f"Expected 12 probability columns, got {len(probability_columns)}"
    )

df["correct"] = (
    df["correct"]
    .astype(str)
    .str.lower()
    .map({"true": True, "false": False})
)

if df["correct"].isna().any():
    raise ValueError("Invalid values found in correct column.")

if df["confidence"].isna().any():
    raise ValueError("Missing confidence values.")

if not df["confidence"].between(0, 1).all():
    raise ValueError("Confidence values must be between 0 and 1.")

print("Validation samples:", len(df))
print("Probability columns:", len(probability_columns))


# =========================================================
# CONFIDENCE SUMMARY
# =========================================================

groups = {
    "overall": df,
    "correct": df[df["correct"]],
    "incorrect": df[~df["correct"]],
}

summary_rows = []

for group_name, group_df in groups.items():
    summary_rows.append({
        "group": group_name,
        "count": len(group_df),
        "mean_confidence": group_df["confidence"].mean(),
        "median_confidence": group_df["confidence"].median(),
        "min_confidence": group_df["confidence"].min(),
        "max_confidence": group_df["confidence"].max(),
    })

wrong = df[~df["correct"]]

summary_rows.append({
    "group": "incorrect_ge_50_percent",
    "count": int((wrong["confidence"] >= 0.50).sum()),
})

summary_rows.append({
    "group": "incorrect_ge_70_percent",
    "count": int((wrong["confidence"] >= 0.70).sum()),
})

summary_rows.append({
    "group": "incorrect_ge_90_percent",
    "count": int((wrong["confidence"] >= 0.90).sum()),
})

summary_df = pd.DataFrame(summary_rows)

summary_path = (
    OUTPUT_DIR / "baseline_confidence_summary.csv"
)

summary_df.to_csv(summary_path, index=False)


# =========================================================
# CALIBRATION BINS AND ECE
# =========================================================

bin_edges = np.linspace(0, 1, NUM_BINS + 1)

bin_rows = []

for i in range(NUM_BINS):

    lower = bin_edges[i]
    upper = bin_edges[i + 1]

    if i == 0:
        mask = (
            (df["confidence"] >= lower)
            & (df["confidence"] <= upper)
        )
    else:
        mask = (
            (df["confidence"] > lower)
            & (df["confidence"] <= upper)
        )

    bin_df = df[mask]

    count = len(bin_df)

    if count > 0:
        accuracy = bin_df["correct"].mean()
        avg_confidence = bin_df["confidence"].mean()
        gap = abs(accuracy - avg_confidence)
        contribution = (count / len(df)) * gap
    else:
        accuracy = np.nan
        avg_confidence = np.nan
        gap = np.nan
        contribution = 0.0

    bin_rows.append({
        "confidence_bin": f"{lower:.1f}-{upper:.1f}",
        "lower_bound": lower,
        "upper_bound": upper,
        "count": count,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "absolute_gap": gap,
        "ece_contribution": contribution,
    })

bins_df = pd.DataFrame(bin_rows)

ece = bins_df["ece_contribution"].sum()

bins_path = (
    OUTPUT_DIR / "baseline_calibration_bins.csv"
)

bins_df.to_csv(bins_path, index=False)


# =========================================================
# CALIBRATION VISUALIZATION
# =========================================================

plot_df = bins_df[bins_df["count"] > 0].copy()

fig, ax = plt.subplots(figsize=(9, 8))

# Ideal calibration line.
ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

# Actual calibration.
ax.plot(
    plot_df["avg_confidence"],
    plot_df["accuracy"],
    marker="o",
    linewidth=2,
    label="CNN Baseline"
)

for _, row in plot_df.iterrows():
    ax.annotate(
        f"n={int(row['count'])}",
        (
            row["avg_confidence"],
            row["accuracy"]
        ),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

ax.set_title(
    f"ColorMate CNN Baseline Calibration\nECE = {ece:.4f}"
)

ax.set_xlabel("Average Confidence")
ax.set_ylabel("Actual Accuracy")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

calibration_path = (
    OUTPUT_DIR / "baseline_calibration.png"
)

plt.savefig(
    calibration_path,
    dpi=180,
    bbox_inches="tight"
)

plt.close(fig)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 65)
print("CONFIDENCE SUMMARY")
print("=" * 65)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}"
    )
)

print("\n" + "=" * 65)
print("CALIBRATION RESULTS")
print("=" * 65)

print(bins_df.to_string(index=False))

print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

print("\nSaved files:")
print(summary_path)
print(bins_path)
print(calibration_path)

print("\nCalibration analysis completed successfully.")