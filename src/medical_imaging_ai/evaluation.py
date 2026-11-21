
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_multilabel(
    y_true,
    y_prob,
    classes,
    threshold=0.5,
):
    """
    Evaluate a multi-label classification model.

    y_true: Actual binary labels, shape (samples, classes)
    y_prob: Predicted sigmoid scores, shape (samples, classes)
    classes: Ordered list of class names
    threshold: Classification threshold
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    results = []

    for i, class_name in enumerate(classes):
        actual = y_true[:, i]
        predicted = y_pred[:, i]
        scores = y_prob[:, i]

        # ROC-AUC requires both positive and negative samples.
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

        results.append({
            "disease": class_name,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "f1": f1_score(
                actual, predicted, zero_division=0
            ),
            "support": int(actual.sum()),
        })

    return pd.DataFrame(results)
