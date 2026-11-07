"""Reuse the recorded train/validation protocol for a frozen product model."""

from pathlib import Path

import numpy as np

from antifraud.infrastructure.csv_data import load_clients, load_operations
from antifraud.infrastructure.model import file_hash, save_model
from antifraud.infrastructure.tabular_model import fit_model, prepare_predictors
from antifraud.infrastructure.temporal import build_features, load_inputs
from antifraud.infrastructure.evaluation import TRAIN_END, VALIDATION_END, choose_thresholds


def train(data_dir, destination):
    data_dir = Path(data_dir)
    warnings = []
    load_clients((data_dir / "clients.csv").read_bytes(), warnings)
    load_operations((data_dir / "transactions_history.csv").read_bytes(), "transactions_history.csv", warnings)
    operations = load_inputs(data_dir)
    # The artifact is fitted and calibrated strictly before October.
    operations = operations[operations["timestamp"] < VALIDATION_END].reset_index(drop=True)
    features = build_features(operations, use_supplied_flags=False)
    known = features["is_fraud"].isin([0, 1])
    train_mask = known & (features["timestamp"] < TRAIN_END)
    valid_mask = known & (features["timestamp"] >= TRAIN_END)
    model, details = fit_model(features, train_mask, valid_mask)
    scores = model.predict_proba(prepare_predictors(features))[valid_mask, 1]
    thresholds = choose_thresholds(features.loc[valid_mask, "is_fraud"].to_numpy(dtype=int), scores)
    if not all(np.isfinite(threshold) for threshold in thresholds.values()):
        raise ValueError("Validation did not produce finite deployable thresholds.")
    metadata = {
        "available_from": VALIDATION_END.isoformat(),
        "default_threshold": thresholds[0.005],
        "thresholds": {str(budget): value for budget, value in thresholds.items()},
        "settings": details,
        "source_sha256": {name: file_hash(data_dir / name)
                          for name in ("clients.csv", "transactions_history.csv")},
        "protocol": "Train August, select parameters and thresholds September; reconstructed novelty flags.",
        "warnings": warnings,
    }
    save_model(model, metadata, destination)
    return metadata
