import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

import graph
import mathematics
import tabular
from features import build_features, load_inputs


TRAIN_END = pd.Timestamp("2026-09-01")
VALIDATION_END = pd.Timestamp("2026-10-01")
TEST_END = pd.Timestamp("2026-10-16")
FALSE_ALARM_BUDGETS = (0.005, 0.01, 0.02)
MODELS = {"tabular": tabular, "graph": graph, "mathematics": mathematics}


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


def run(data_dir):
    operations = load_inputs(data_dir)
    operations = operations.loc[operations["timestamp"] < TEST_END].reset_index(drop=True)
    masks = chronological_masks(operations)
    labels = operations["is_fraud"]
    model_input = hide_evaluation_labels(operations, masks)
    validation_labels = labels[masks["validation"]].to_numpy(dtype=int)
    frozen = {}

    for variant, use_supplied in [("provided_flags", True), ("reconstructed_flags", False)]:
        features = build_features(model_input, use_supplied_flags=use_supplied)
        for name, model in MODELS.items():
            if name == "graph" and not use_supplied:
                continue  # The graph model only reads transfer and account graph inputs.
            scores, settings = model.fit_predict(features, masks["train"], masks["validation"])
            scores = np.asarray(scores, dtype=float)
            if scores.shape != (len(operations),) or not np.isfinite(scores).all():
                raise ValueError(f"{name} must return one finite score per operation.")
            thresholds = choose_thresholds(validation_labels, scores[masks["validation"]])
            frozen[f"{name}/{variant}" if name != "graph" else name] = (scores, settings, thresholds)

    report = {
        "periods": {
            name: {
                "labelled": int(mask.sum()),
                "positive": int(labels[mask].sum()),
                "prevalence": float(labels[mask].mean()),
            }
            for name, mask in masks.items()
        },
        "models": {},
    }
    for name, (scores, settings, thresholds) in frozen.items():
        model_result = {"settings": settings}
        for period, mask in masks.items():
            model_result[period] = evaluate_scores(labels[mask].to_numpy(dtype=int), scores[mask], thresholds)
        report["models"][name] = model_result
    return report


def main():
    parser = argparse.ArgumentParser(description="Compare three fraud models using labelled historical operations.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    report = run(args.data_dir)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, model in report["models"].items():
        validation = model["validation"]
        test = model["test"]
        point = test["operating_points"]["0.01"]
        print(f"{name:32s} AP Sep {validation['average_precision']:.4f}  AP Oct {test['average_precision']:.4f}"
              f"  Oct caught {point['tp']:3d}  false alerts {point['fp']:3d}  FPR {point['fpr']:.2%}")


if __name__ == "__main__":
    main()
