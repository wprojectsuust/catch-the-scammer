import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from antifraud.application import AnalyzeOperations
from antifraud.infrastructure.feature_engine import CausalFeatureEngine
from antifraud.infrastructure.model import CatBoostRiskModel, save_model
from antifraud.infrastructure.tabular_model import CATEGORICAL, prepare_predictors
from antifraud.sandbox import SCENARIOS, run_scenarios, scenario_dataset


class SandboxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temporary.name) / "candidate"
        cls.available_from = datetime(2026, 10, 1)
        engine = CausalFeatureEngine()
        rows = [engine.prepare(scenario_dataset(name, cls.available_from))[0] for name in SCENARIOS]
        predictors = prepare_predictors(pd.DataFrame([row.values for row in rows]))
        cls.fitted = CatBoostClassifier(iterations=10, depth=2, random_seed=42, thread_count=1,
                                       verbose=False, allow_writing_files=False, cat_features=list(CATEGORICAL))
        cls.fitted.fit(predictors, [0, 0, 1, 1, 0, 0, 0])
        save_model(cls.fitted, {"available_from": cls.available_from.isoformat(), "default_threshold": 0.5}, cls.path)
        cls.model = CatBoostRiskModel(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_all_seven_scenarios_and_native_explanations(self):
        report = run_scenarios(self.model)
        self.assertTrue(report["passed"], report)
        self.assertEqual(len(report["scenarios"]), 7)

    def test_saved_model_matches_original_model(self):
        rows = CausalFeatureEngine().prepare(scenario_dataset("new_device", self.available_from))
        expected = self.fitted.predict_proba(prepare_predictors(pd.DataFrame([row.values for row in rows])))[0, 1]
        self.assertAlmostEqual(self.model.predict(rows)[0].score, expected, places=12)

    def test_one_and_batch_predictions_agree(self):
        rows = [CausalFeatureEngine().prepare(scenario_dataset(name, self.available_from))[0] for name in SCENARIOS]
        batched = self.model.predict(rows)
        np.testing.assert_allclose([self.model.predict([row])[0].score for row in rows],
                                   [prediction.score for prediction in batched], rtol=1e-12)

    def test_unknown_customer_is_supported_by_real_model(self):
        from dataclasses import replace
        dataset = replace(scenario_dataset("missing", self.available_from), clients=())
        analysis = AnalyzeOperations(CausalFeatureEngine(), self.model).execute(dataset)
        self.assertTrue(math.isfinite(analysis.assessments[0].prediction.score))
