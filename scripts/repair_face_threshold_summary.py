from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "dataset_audit"

baseline_path = OUTPUT_DIR / "face_detection_audit.csv"
threshold_path = OUTPUT_DIR / "face_detection_audit_065.csv"

if not baseline_path.exists() or not threshold_path.exists():
    raise FileNotFoundError("File audit baseline atau threshold 0.65 tidak ditemukan.")

baseline = pd.read_csv(baseline_path)
new_df = pd.read_csv(threshold_path)

keys = ["split", "image_index"]

comparison = baseline.merge(
    new_df,
    on=keys,
    how="outer",
    suffixes=("_050", "_065"),
    indicator=True,
)

comparison["face_count_change"] = (
    comparison["face_count_065"] -
    comparison["face_count_050"]
)

comparison_path = OUTPUT_DIR / "face_detection_comparison_050_065.csv"

comparison.to_csv(
    comparison_path,
    index=False,
    encoding="utf-8-sig",
)


def summarize(df, suffix):
    valid = df[df[f"status{suffix}"] == "success"].copy()

    result = (
        valid.groupby(
            ["split", f"label{suffix}"],
            dropna=False,
        )
        .agg(
            images=("image_index", "count"),
            no_face=(f"face_count{suffix}", lambda x: (x == 0).sum()),
            single_face=(f"face_count{suffix}", lambda x: (x == 1).sum()),
            multiple_faces=(f"face_count{suffix}", lambda x: (x > 1).sum()),
            mean_confidence=(f"max_confidence{suffix}", "mean"),
        )
        .reset_index()
        .rename(columns={
            f"label{suffix}": "label"
        })
    )

    return result


summary_050 = summarize(comparison, "_050")
summary_065 = summarize(comparison, "_065")

summary = summary_050.merge(
    summary_065,
    on=["split", "label"],
    suffixes=("_050", "_065"),
)

for metric in ["no_face", "single_face", "multiple_faces"]:
    summary[f"{metric}_change"] = (
        summary[f"{metric}_065"] - summary[f"{metric}_050"]
    )

summary_path = OUTPUT_DIR / "face_detection_summary_050_065.csv"

summary.to_csv(
    summary_path,
    index=False,
    encoding="utf-8-sig",
)

print("=" * 60)
print("FACE DETECTION SUMMARY REPAIRED")
print("=" * 60)

print("\nTotal baseline:", len(baseline))
print("Total threshold 0.65:", len(new_df))
print("Comparison rows:", len(comparison))

print("\nProcessing errors at 0.65:",
      (new_df["status"] != "success").sum())

print("\nSummary by split and label:")
print(
    summary[
        [
            "split",
            "label",
            "single_face_050",
            "single_face_065",
            "multiple_faces_050",
            "multiple_faces_065",
            "no_face_050",
            "no_face_065",
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(comparison_path)
print(summary_path)