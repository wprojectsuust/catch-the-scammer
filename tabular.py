"""Compatibility entry point for the original CatBoost experiments."""

from antifraud.infrastructure.tabular_model import CANDIDATES, CATEGORICAL, COMMON_PARAMS, NUMERIC, fit_predict

__all__ = ["CANDIDATES", "CATEGORICAL", "COMMON_PARAMS", "NUMERIC", "fit_predict"]
