from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# Lokasi file
ROOT_DIR = Path(__file__).resolve().parents[1]
HISTORY_PATH = ROOT_DIR / "reports" / "dataset_audit" / "baseline_training_history.csv"
OUTPUT_DIR = ROOT_DIR / "reports" / "dataset_audit"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Membaca riwayat training
df = pd.read_csv(HISTORY_PATH)

epochs = df["epoch"]


def plot_metric(train_column, val_column, title, ylabel, filename):
    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        df[train_column],
        marker="o",
        label="Training"
    )

    plt.plot(
        epochs,
        df[val_column],
        marker="o",
        label="Validation"
    )

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(epochs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()

    print(f"Saved: {output_path}")


# 1. Loss
plot_metric(
    "train_loss",
    "val_loss",
    "ColorMate CNN - Training vs Validation Loss",
    "Loss",
    "baseline_loss_curve.png"
)

# 2. Accuracy
plot_metric(
    "train_accuracy",
    "val_accuracy",
    "ColorMate CNN - Training vs Validation Accuracy",
    "Accuracy",
    "baseline_accuracy_curve.png"
)

# 3. Macro F1
plot_metric(
    "train_macro_f1",
    "val_macro_f1",
    "ColorMate CNN - Training vs Validation Macro F1",
    "Macro F1 Score",
    "baseline_macro_f1_curve.png"
)

print("\nAll training curves generated successfully.")