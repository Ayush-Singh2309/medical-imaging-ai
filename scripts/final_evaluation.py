
from pathlib import Path

import numpy as np
import pandas as pd
import keras

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from medical_imaging_ai.constants import CLASSES
from medical_imaging_ai.preprocessing import create_dataset


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data/raw/NIH_ChestXray14/images"
VAL_CSV = ROOT / "data/splits/validation.csv"
TEST_CSV = ROOT / "data/splits/test.csv"
MODEL_PATH = ROOT / "models/best_model.keras"

OUTPUT_DIR = ROOT / "data/processed/evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

available_images = {
    p.name for p in IMAGE_DIR.glob("*.png")
}


def load_split(csv_path):

    df = pd.read_csv(csv_path)

    df = df[
        df["image_index"].isin(available_images)
    ].copy()

    if df.empty:
        raise ValueError(f"No available images in {csv_path}")

    return df


val_df = load_split(VAL_CSV)
test_df = load_split(TEST_CSV)

print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))


# --------------------------------------------------
# Create TensorFlow datasets
# --------------------------------------------------

val_ds = create_dataset(
    dataframe=val_df,
    image_dir=IMAGE_DIR,
    batch_size=BATCH_SIZE,
    training=False,
)

test_ds = create_dataset(
    dataframe=test_df,
    image_dir=IMAGE_DIR,
    batch_size=BATCH_SIZE,
    training=False,
)


# --------------------------------------------------
# Load model and generate predictions
# --------------------------------------------------

model = keras.models.load_model(MODEL_PATH)

y_val = val_df[CLASSES].to_numpy(dtype=np.float32)
y_test = test_df[CLASSES].to_numpy(dtype=np.float32)

val_prob = model.predict(val_ds, verbose=1)
test_prob = model.predict(test_ds, verbose=1)

assert val_prob.shape == y_val.shape
assert test_prob.shape == y_test.shape


# --------------------------------------------------
# Select thresholds using validation data only
# --------------------------------------------------

thresholds = []

for i, disease in enumerate(CLASSES):

    actual = y_val[:, i]
    scores = val_prob[:, i]

    # Cannot optimize a threshold without both classes
    if len(np.unique(actual)) < 2:
        threshold = 0.5

    else:
        precision, recall, candidate_thresholds = (
            precision_recall_curve(actual, scores)
        )

        f1_values = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-8)
        )

        best_index = np.argmax(f1_values)
        threshold = candidate_thresholds[best_index]

    thresholds.append(float(threshold))

threshold_df = pd.DataFrame({
    "disease": CLASSES,
    "threshold": thresholds,
})

threshold_df.to_csv(
    OUTPUT_DIR / "optimal_thresholds.csv",
    index=False,
)


# --------------------------------------------------
# Final evaluation on test data
# --------------------------------------------------

results = []

for i, disease in enumerate(CLASSES):

    actual = y_test[:, i]
    scores = test_prob[:, i]
    threshold = thresholds[i]

    predicted = (scores >= threshold).astype(int)

    roc_auc = (
        roc_auc_score(actual, scores)
        if len(np.unique(actual)) == 2
        else np.nan
    )

    pr_auc = (
        average_precision_score(actual, scores)
        if actual.sum() > 0
        else np.nan
    )

    precision_value = precision_score(
        actual, predicted, zero_division=0
    )

    recall_value = recall_score(
        actual, predicted, zero_division=0
    )

    f1_value = f1_score(
        actual, predicted, zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    results.append({
        "disease": disease,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision_value,
        "recall": recall_value,
        "f1": f1_value,
        "support": int(actual.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "final_test_metrics.csv",
    index=False,
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nOPTIMAL VALIDATION THRESHOLDS")
print(threshold_df.to_string(index=False))

print("\nFINAL TEST RESULTS")
print(results_df.to_string(index=False))

metric_columns = [
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
]

print("\nMACRO AVERAGES")
print(results_df[metric_columns].mean())

print("\nEvaluation files saved in:", OUTPUT_DIR)
