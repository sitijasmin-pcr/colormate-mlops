from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data/processed/splits/armocromia_split_seed42.csv"
OUTPUT = ROOT / "reports/dataset_audit/background_perturbation"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

validation = df[df["split"] == "validation"].copy()

samples = (
    validation
    .groupby("label", group_keys=False)
    .sample(n=5, random_state=42)
    .sort_values(["label", "image_index"])
    .reset_index(drop=True)
)

if len(samples) != 60:
    raise ValueError(f"Expected 60 samples, got {len(samples)}")

if samples["label"].nunique() != 12:
    raise ValueError("Expected all 12 classes.")

samples.to_csv(
    OUTPUT / "background_test_samples.csv",
    index=False
)

print("=" * 60)
print("COLORMATE BACKGROUND TEST SAMPLE SELECTION")
print("=" * 60)

print("Total samples:", len(samples))
print("\nSamples per class:")
print(samples["label"].value_counts().sort_index().to_string())

print("\nSaved:")
print(OUTPUT / "background_test_samples.csv")