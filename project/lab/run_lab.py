import argparse
import json
from pathlib import Path

import numpy as np

import graph
import mathematics
import tabular
from features import build_features, load_inputs


from antifraud.infrastructure.evaluation import (
    TRAIN_END, VALIDATION_END, TEST_END, FALSE_ALARM_BUDGETS,
    chronological_masks, hide_evaluation_labels, choose_thresholds, measure, evaluate_scores,
)

MODELS = {"tabular": tabular, "graph": graph, "mathematics": mathematics}


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
