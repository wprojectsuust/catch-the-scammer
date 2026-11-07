"""Evaluate the two MVP candidates on sequential historical windows."""

import argparse
import json
from pathlib import Path

import pandas as pd

import mathematics
import tabular
from features import build_features, load_inputs
from run_lab import choose_thresholds, chronological_masks, evaluate_scores, hide_evaluation_labels


# The three evaluation periods are disjoint. Each model is selected again using
# only the validation period that precedes its own evaluation period.
WINDOWS = (
    ("2026-08-16", "2026-09-01", "2026-09-16"),
    ("2026-09-01", "2026-09-16", "2026-10-01"),
    ("2026-09-01", "2026-10-01", "2026-10-16"),
)
MODELS = {"catboost": tabular, "mathematics": mathematics}


def evaluate_window(operations, train_end, validation_end, test_end):
    masks = chronological_masks(operations, train_end, validation_end, test_end)
    labels = operations["is_fraud"]
    features = build_features(hide_evaluation_labels(operations, masks), use_supplied_flags=False)
    validation = masks["validation"]
    frozen = {}
    for name, model in MODELS.items():
        scores, settings = model.fit_predict(features, masks["train"], validation)
        thresholds = choose_thresholds(labels[validation].to_numpy(dtype=int), scores[validation])
        frozen[name] = (scores, settings, thresholds)

    result = {
        "end_exclusive": {
            "train": train_end.date().isoformat(),
            "validation": validation_end.date().isoformat(),
            "test": test_end.date().isoformat(),
        },
        "counts": {
            period: {"labelled": int(mask.sum()), "positive": int(labels[mask].sum())}
            for period, mask in masks.items()
        },
        "models": {},
    }
    for name, (scores, settings, thresholds) in frozen.items():
        result["models"][name] = {"settings": settings}
        for period, mask in masks.items():
            actual = labels[mask].to_numpy(dtype=int)
            result["models"][name][period] = evaluate_scores(actual, scores[mask], thresholds)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("backtest-results.json"))
    args = parser.parse_args()
    operations = load_inputs(args.data_dir)
    results = []
    for train_end, validation_end, test_end in (tuple(map(pd.Timestamp, dates)) for dates in WINDOWS):
        available = operations.loc[operations["timestamp"] < test_end]
        result = evaluate_window(available, train_end, validation_end, test_end)
        results.append(result)
        for name, model in result["models"].items():
            validation_ap = model["validation"]["average_precision"]
            test_ap = model["test"]["average_precision"]
            test_point = model["test"]["operating_points"]["0.01"]
            print(f"{validation_end.date()} {name:11s} AP val {validation_ap:.3f} "
                  f"AP test {test_ap:.3f} TP {test_point['tp']} FP {test_point['fp']}", flush=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
