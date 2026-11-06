import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import evaluate_window
from features import build_features, load_inputs
from graph import build_graph_features
from run_lab import choose_thresholds, chronological_masks, hide_evaluation_labels, measure


DATA_DIR = Path(__file__).resolve().parents[2] / "work" / "hackathon" / "data"


class CausalityTest(unittest.TestCase):
    def test_operations_at_same_time_do_not_see_each_other(self):
        operations = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-08-01 12:00", "2026-08-01 12:00", "2026-08-01 12:01",
                                         "2026-08-01 12:02"]),
            "sender_account_id": ["A"] * 4,
            "recipient_account_id": ["B"] * 4,
            "device_id": ["device"] * 4,
            "region": ["region"] * 4,
            "amount_rub": [100] * 4,
            "balance_before_rub": [1_000] * 4,
            "registration_date": pd.to_datetime(["2026-01-01"] * 4),
            "monthly_income_rub": [10_000] * 4,
            "transaction_type": ["перевод"] * 4,
            "status": ["исполнена", "исполнена", "отклонена", "исполнена"],
        })
        features = build_features(operations, use_supplied_flags=False)
        self.assertEqual(features["prior_count"].tolist(), [0, 0, 2, 3])
        self.assertEqual(features["device_seen"].tolist(), [0, 0, 1, 1])
        graph_features = build_graph_features(operations)
        self.assertEqual(graph_features["direct_edge_count"].tolist(), [0, 0, 2, 2])

    @unittest.skipUnless(DATA_DIR.is_dir(), "Requires the supplied historical CSV files.")
    def test_removing_october_keeps_september_features(self):
        operations = load_inputs(DATA_DIR)
        past = operations[operations["timestamp"] < pd.Timestamp("2026-10-01")]
        for extractor in (build_features, build_graph_features):
            with self.subTest(extractor=extractor.__name__):
                pd.testing.assert_frame_equal(extractor(operations).iloc[:len(past)], extractor(past))

    def test_threshold_handles_ties_at_budget_boundary(self):
        labels = np.array([1, 1, *([0] * 100)])
        scores = np.array([0.9, 0.8, 0.9, 0.8, *([0.1] * 98)])
        thresholds = choose_thresholds(labels, scores)
        self.assertTrue(np.isinf(thresholds[0.005]))
        self.assertEqual(measure(labels, scores, thresholds[0.01])["tp"], 1)
        self.assertEqual(measure(labels, scores, thresholds[0.02])["fp"], 2)

    def test_split_boundaries_and_unknown_labels(self):
        operations = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-30", periods=9, freq="D"),
            "is_fraud": [0, 1, 0, 1, None, 0, 1, 1, 0],
        })
        dates = tuple(map(pd.Timestamp, ("2026-09-01", "2026-09-04", "2026-09-06")))
        masks = chronological_masks(operations, *dates)
        self.assertEqual({name: list(mask[mask].index) for name, mask in masks.items()},
                         {"train": [0, 1], "validation": [2, 3], "test": [5, 6]})
        visible = hide_evaluation_labels(operations, masks)["is_fraud"]
        self.assertEqual(visible.dropna().index.tolist(), [0, 1, 2, 3])
        self.assertEqual(operations.loc[6, "is_fraud"], 1)

    def test_swapping_evaluation_labels_changes_metrics_without_changing_model_selection(self):
        operations = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-01", periods=30, freq="D"),
            "is_fraud": [0, 1] * 15,
            "amount_rub": [100, 900] * 15,
            "balance_before_rub": 1_000,
            "registration_date": pd.Timestamp("2026-01-01"),
            "monthly_income_rub": 10_000,
            "sender_account_id": "A",
            "recipient_account_id": "B",
            "device_id": "device",
            "region": "region",
            "city": "city",
            "account_type": "type",
            "client_category": "category",
            "transaction_type": "перевод",
        })
        dates = tuple(map(pd.Timestamp, ("2026-08-11", "2026-08-21", "2026-08-31")))
        original = evaluate_window(operations, *dates)
        test_rows = operations["timestamp"] >= dates[1]
        operations.loc[test_rows, "is_fraud"] = 1 - operations.loc[test_rows, "is_fraud"]
        changed = evaluate_window(operations, *dates)
        for name, model in original["models"].items():
            with self.subTest(model=name):
                changed_model = changed["models"][name]
                for key in ("settings", "train", "validation"):
                    self.assertEqual(model[key], changed_model[key])
                self.assertNotEqual(model["test"]["average_precision"], changed_model["test"]["average_precision"])


if __name__ == "__main__":
    unittest.main()
