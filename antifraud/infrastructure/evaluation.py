"""Shared chronological evaluation protocol for experiments and training."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

TRAIN_END = pd.Timestamp("2026-09-01")
VALIDATION_END = pd.Timestamp("2026-10-01")
TEST_END = pd.Timestamp("2026-10-16")
FALSE_ALARM_BUDGETS = (0.005, 0.01, 0.02)


def chronological_masks(operations, train_end=TRAIN_END, validation_end=VALIDATION_END, test_end=TEST_END):
    if not train_end < validation_end < test_end:
        raise ValueError("Chronological split boundaries must be strictly increasing.")
    timestamp = operations["timestamp"]
    known = operations["is_fraud"].isin([0, 1])
    masks = {
        "train": known & (timestamp < train_end),
        "validation": known & (timestamp >= train_end) & (timestamp < validation_end),
        "test": known & (timestamp >= validation_end) & (timestamp < test_end),
    }
    for name, mask in masks.items():
        if operations.loc[mask, "is_fraud"].nunique() != 2:
            raise ValueError(f"Both classes are required in the {name} period.")
    return masks


def hide_evaluation_labels(operations, masks):
    return operations.assign(is_fraud=operations["is_fraud"].where(masks["train"] | masks["validation"]))


def choose_thresholds(labels, scores):
    fpr, recall, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    choices = {}
    for budget in FALSE_ALARM_BUDGETS:
        eligible = np.flatnonzero(fpr <= budget)
        # ROC thresholds descend, so ties in recall select the stricter threshold.
        best = eligible[np.argmax(recall[eligible])]
        choices[budget] = float(thresholds[best])
    return choices


def measure(labels, scores, threshold):
    positives = labels == 1
    alerts = scores >= threshold
    tp = int(np.count_nonzero(alerts & positives))
    fp = int(np.count_nonzero(alerts & ~positives))
    fn = int(np.count_nonzero(~alerts & positives))
    tn = int(np.count_nonzero(~alerts & ~positives))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / (tp + fn),
        "precision": tp / (tp + fp) if tp + fp else None,
        "fpr": fp / (fp + tn),
    }


def evaluate_scores(labels, scores, thresholds):
    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "operating_points": {
            str(budget): {
                "threshold": threshold if np.isfinite(threshold) else None,
                **measure(labels, scores, threshold),
            }
            for budget, threshold in thresholds.items()
        },
    }


