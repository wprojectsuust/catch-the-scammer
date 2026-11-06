"""Versioned CatBoost artifacts and explanations on the raw log-odds scale."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from antifraud.domain import Contribution, InvalidData, Prediction
from antifraud.infrastructure.tabular_model import CATEGORICAL, prepare_predictors


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CatBoostRiskModel:
    def __init__(self, directory):
        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if self.metadata["schema_version"] != 1:
            raise InvalidData("Неподдерживаемая версия модели.")
        if file_hash(directory / "model.cbm") != self.metadata["model_sha256"]:
            raise InvalidData("Контрольная сумма модели не совпала.")
        for filename, digest in self.metadata["feature_sources"].items():
            if file_hash(Path(__file__).parent / filename) != digest:
                raise InvalidData("Код подготовки признаков изменился. Требуется повторная проверка и обучение модели.")
        self.available_from = datetime.fromisoformat(self.metadata["available_from"])
        self.default_threshold = self.metadata["default_threshold"]
        self.model = CatBoostClassifier()
        self.model.load_model(str(directory / "model.cbm"))

    def predict(self, rows):
        if not rows:
            return ()
        predictors = prepare_predictors(pd.DataFrame([row.values for row in rows]))
        pool = Pool(predictors, cat_features=list(CATEGORICAL))
        probabilities = self.model.predict_proba(pool, thread_count=2)[:, 1]
        contributions = self.model.get_feature_importance(pool, type="ShapValues", thread_count=2)
        return tuple(
            Prediction(
                float(score), float(shap[-1]),
                tuple(Contribution(column, row.values[column], float(value))
                      for column, value in zip(predictors.columns, shap[:-1])),
            )
            for score, shap, row in zip(probabilities, contributions, rows)
        )


def save_model(model, metadata, destination):
    destination = Path(destination)
    if destination.exists():
        raise InvalidData("Каталог модели уже существует. Выберите новый каталог версии.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix=".candidate-") as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        model.save_model(str(staging / "model.cbm"))
        metadata = {
            **metadata,
            "schema_version": 1,
            "model_sha256": file_hash(staging / "model.cbm"),
            "feature_sources": {
                name: file_hash(Path(__file__).parent / name)
                for name in ("temporal.py", "feature_engine.py", "tabular_model.py")
            },
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        os.rename(staging, destination)
