"""Regularized CatBoost baseline on tabular and behavioral features."""

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score

CATEGORICAL = (
    "transaction_type",
    "region",
    "city",
    "account_type",
    "client_category",
)
NUMERIC = (
    "amount_log",
    "balance_fraction",
    "hour_sin",
    "hour_cos",
    "is_new_device",
    "is_new_recipient",
    "account_age_days",
    "income_log",
    "prior_count",
    "amount_robust_z",
    "velocity_5m",
    "velocity_60m",
    "velocity_24h",
    "region_seen",
    "device_seen",
    "recipient_seen",
    "geo_change",
)

# In an exact tie, prefer the first, smallest tree ensemble.
CANDIDATES = (
    {"depth": 3, "iterations": 200},
    {"depth": 4, "iterations": 350},
    {"depth": 5, "iterations": 350},
)
COMMON_PARAMS = {
    "learning_rate": 0.05,
    "l2_leaf_reg": 10.0,
    "loss_function": "Logloss",
    "random_seed": 42,
    "thread_count": 2,
    "verbose": False,
    "allow_writing_files": False,
}


def _confirmed(features, mask):
    rows = features.loc[mask, "is_fraud"]
    labels = rows.dropna().astype(float)
    if set(labels.unique()) != {0.0, 1.0}:
        raise ValueError("Each fitting and tuning split must contain both confirmed classes.")
    positions = np.flatnonzero(np.asarray(mask, dtype=bool))[rows.notna().to_numpy()]
    return positions, labels.to_numpy(dtype=int)


def prepare_predictors(features):
    predictors = features.loc[:, list(NUMERIC + CATEGORICAL)].copy()
    for column in NUMERIC:
        predictors[column] = predictors[column].astype(float)
    for column in CATEGORICAL:
        predictors[column] = predictors[column].astype("string").fillna("__missing__").astype(str)

    return predictors


def fit_model(features, train_mask, validation_mask):
    """Fit on training rows and select by validation AP."""
    predictors = prepare_predictors(features)
    train_rows, train_labels = _confirmed(features, train_mask)
    valid_rows, valid_labels = _confirmed(features, validation_mask)
    if np.intersect1d(train_rows, valid_rows).size:
        raise ValueError("Training and validation rows must be disjoint.")

    best_model = None
    best_ap = -1.0
    comparison = []
    for params in CANDIDATES:
        model = CatBoostClassifier(**COMMON_PARAMS, **params, cat_features=list(CATEGORICAL))
        model.fit(predictors.iloc[train_rows], train_labels)
        valid_scores = model.predict_proba(predictors.iloc[valid_rows])[:, 1]
        ap = float(average_precision_score(valid_labels, valid_scores))
        comparison.append({"params": dict(params), "validation_ap": ap})
        if ap > best_ap:
            best_model, best_params, best_ap = model, params, ap

    return best_model, {
        "model": "CatBoost",
        "params": {**COMMON_PARAMS, **best_params},
        "validation_ap": best_ap,
        "validation_comparison": comparison,
    }


def fit_predict(features, train_mask, validation_mask):
    model, details = fit_model(features, train_mask, validation_mask)
    return model.predict_proba(prepare_predictors(features))[:, 1], details
