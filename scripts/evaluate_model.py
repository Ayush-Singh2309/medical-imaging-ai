
from pathlib import Path

import numpy as np
import pandas as pd
import keras

from medical_imaging_ai.constants import CLASSES
from medical_imaging_ai.preprocessing import create_dataset
from medical_imaging_ai.evaluation import evaluate_multilabel


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = PROJECT_ROOT / "data/splits/test.csv"
IMAGE_DIR = PROJECT_ROOT / "data/raw/NIH_ChestXray14/images"
MODEL_PATH = PROJECT_ROOT / "models/best_model.keras"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/model_evaluation.csv"

BATCH_SIZE = 32


# 1. Load test split
test_df = pd.read_csv(TEST_CSV)

# Keep only images available locally
available_images = {
    path.name for path in IMAGE_DIR.glob("*.png")
}

test_df = test_df[
    test_df["image_index"].isin(available_images)
].copy()

if test_df.empty:
    raise ValueError("No test images found.")

# Ensure all required labels exist
missing_classes = set(CLASSES) - set(test_df.columns)

if missing_classes:
    raise ValueError(f"Missing label columns: {missing_classes}")

print("Test samples:", len(test_df))


# 2. Create test dataset
test_ds = create_dataset(
    dataframe=test_df,
    image_dir=IMAGE_DIR,
    batch_size=BATCH_SIZE,
    training=False
)


# 3. Extract actual labels
y_true = test_df[CLASSES].to_numpy(dtype=np.float32)


# 4. Load model
model = keras.models.load_model(MODEL_PATH)


# 5. Generate predictions
y_prob = model.predict(test_ds, verbose=1)


# Inspect prediction score distributions

for i, disease in enumerate(CLASSES):

    scores = y_prob[:, i]
    actual = y_true[:, i]

    positive_scores = scores[actual == 1]
    negative_scores = scores[actual == 0]

    print(f"\n{disease}")
    print("-" * 45)

    print("Minimum:", scores.min())
    print("Maximum:", scores.max())
    print("Mean:", scores.mean())

    print("Positive sample mean:", positive_scores.mean())
    print("Negative sample mean:", negative_scores.mean())

    print(
        "Percentiles:",
        np.percentile(scores, [50, 75, 90, 95, 99])
    )

    print("Predictions >= 0.5:", (scores >= 0.5).sum())


assert y_true.shape == y_prob.shape, (
    f"Shape mismatch: y_true={y_true.shape}, "
    f"y_prob={y_prob.shape}"
)


# 6. Evaluate model
results = evaluate_multilabel(
    y_true=y_true,
    y_prob=y_prob,
    classes=CLASSES,
    threshold=0.5
)


# 7. Display results
print("\nPer-class evaluation:")
print(results.to_string(index=False))

print("\nMacro averages:")
print(
    results[
        ["roc_auc", "pr_auc", "precision", "recall", "f1"]
    ].mean()
)


# 8. Save results
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(OUTPUT_PATH, index=False)

print(f"\nResults saved to: {OUTPUT_PATH}")
