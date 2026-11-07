"""Bounded additive risk score with explicit per-transaction contributions."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


TERM_FORMULAS = {
    "large_amount": "clip(amount_robust_z / 4, 0, 1)",
    "balance_pressure": "clip(balance_fraction, 0, 1)",
    "burst": "max(clip(velocity_5m / 3, 0, 1), clip(velocity_60m / 6, 0, 1))",
    "new_device": "max(is_new_device, n / (n + 10) × (1 - device_seen))",
    "new_recipient": "n / (n + 10) × (1 - recipient_seen)",
    "unusual_region": "n / (n + 10) × (1 - region_seen)",
    "combined_signal": "large_amount × new_device",
}

WEIGHT_PROFILES = {
    "balanced": dict(zip(TERM_FORMULAS, (0.24, 0.14, 0.16, 0.26, 0.03, 0.09, 0.08))),
    "device": dict(zip(TERM_FORMULAS, (0.16, 0.10, 0.10, 0.42, 0.02, 0.06, 0.14))),
    "behavior": dict(zip(TERM_FORMULAS, (0.30, 0.17, 0.22, 0.15, 0.02, 0.10, 0.04))),
}


def _risk_terms(features):
    prior_count = features["prior_count"].fillna(0).clip(lower=0)
    history_weight = prior_count / (prior_count + 10)
    large_amount = (features["amount_robust_z"].fillna(0) / 4).clip(0, 1)
    device_unseen = 1 - features["device_seen"].fillna(1).astype(float)
    new_device = np.maximum(
        features["is_new_device"].fillna(0).astype(float).clip(0, 1),
        history_weight * device_unseen,
    )
    burst = np.maximum(
        (features["velocity_5m"].fillna(0) / 3).clip(0, 1),
        (features["velocity_60m"].fillna(0) / 6).clip(0, 1),
    )
    return pd.DataFrame(
        {
            "large_amount": large_amount,
            "balance_pressure": features["balance_fraction"].fillna(0).clip(0, 1),
            "burst": burst,
            "new_device": new_device,
            "new_recipient": history_weight
            * (1 - features["recipient_seen"].fillna(1).astype(float)),
            "unusual_region": history_weight
            * (1 - features["region_seen"].fillna(1).astype(float)),
            "combined_signal": large_amount * new_device,
        },
        index=features.index,
    )


def contributions(features, profile):
    """Return weighted terms. The risk score is their row-wise sum."""
    return _risk_terms(features).mul(pd.Series(WEIGHT_PROFILES[profile]), axis="columns")


def fit_predict(features, train_mask, validation_mask):
    """Select a fixed weight profile by validation AP and score every row."""
    # The heuristic profiles are fixed before seeing any labels. Validation
    # labels select a profile and the training labels are unused.
    validation_mask = np.asarray(validation_mask, dtype=bool)
    validation_labels = features.loc[validation_mask, "is_fraud"].astype(int)
    terms = _risk_terms(features)
    validation_scores = {
        profile: float(
            average_precision_score(
                validation_labels,
                terms.loc[validation_mask].mul(pd.Series(weights), axis="columns").sum(axis=1),
            )
        )
        for profile, weights in WEIGHT_PROFILES.items()
    }
    selected = max(validation_scores, key=validation_scores.get)
    weights = WEIGHT_PROFILES[selected]
    scores = terms.mul(pd.Series(weights), axis="columns").sum(axis=1)
    return scores.to_numpy(), {
        "architecture": "Bounded additive risk score",
        "profile": selected,
        "weights": weights,
        "formulas": TERM_FORMULAS,
        "validation_ap_by_profile": validation_scores,
    }
