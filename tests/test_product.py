import csv
import io
import unittest
from dataclasses import replace
from datetime import datetime
from unittest.mock import Mock

from antifraud.application import AnalyzeOperations
from antifraud.domain import Client, Dataset, InvalidData, Operation, Prediction
from antifraud.infrastructure.csv_data import load_dataset
from antifraud.infrastructure.feature_engine import CausalFeatureEngine
from antifraud.interfaces.presentation import export_csv, valid_threshold


def sample_operation(identity="new", **changes):
    values = dict(
        transaction_id=identity, sender_account_id="A", recipient_account_id="B", customer_id="C",
        timestamp=datetime(2026, 10, 16, 12), amount_rub=100, balance_before_rub=1000,
        transaction_type="перевод", device_id="D", region="R", city="City", status="исполнена",
    )
    values.update(changes)
    return Operation(**values)


def sample_dataset():
    client = Client("C", "A", datetime(2026, 1, 1), "дебетовый", "студент", 20000)
    return Dataset((client,), (), (sample_operation(),))


def csv_bytes(rows, fields):
    output = io.StringIO()
    writer = csv.DictWriter(output, fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


CLIENTS = b"client_id,account_id\nC,A\n"
FIELDS = ["transaction_id", "sender_account_id", "recipient_account_id", "customer_id",
          "timestamp", "amount_rub", "transaction_type", "status", "currency"]


def csv_operation(identity="new", **changes):
    row = dict(zip(FIELDS, [identity, "A", "B", "C", "2026-10-16 12:00:00", "100",
                           "перевод", "исполнена", "RUB"]))
    row.update(changes)
    return row


class CsvTests(unittest.TestCase):
    def load(self, rows, history=()):
        return load_dataset(CLIENTS, csv_bytes(history, FIELDS), csv_bytes(rows, FIELDS))

    def test_optional_columns_and_exact_duplicates(self):
        data = self.load([csv_operation(), csv_operation()])
        self.assertEqual(len(data.current), 1)
        self.assertIsNone(data.current[0].balance_before_rub)
        self.assertTrue(data.warnings)

    def test_conflicting_duplicates_and_cross_file_ids(self):
        with self.assertRaises(InvalidData):
            self.load([csv_operation(), csv_operation(amount_rub="200")])
        with self.assertRaises(InvalidData):
            self.load([csv_operation()], [csv_operation()])

    def test_invalid_amount_currency_and_timestamp(self):
        for change in ({"amount_rub": "NaN"}, {"amount_rub": "inf"}, {"amount_rub": "-1"},
                       {"currency": "USD"}, {"timestamp": "bad"}, {"timestamp": "2026-10-16T12:00:00+03:00"}):
            with self.subTest(change=change), self.assertRaises(InvalidData):
                self.load([csv_operation(**change)])

    def test_schema_and_account_mismatch(self):
        with self.assertRaises(InvalidData):
            load_dataset(CLIENTS, b"timestamp\n", b"timestamp\n")
        with self.assertRaises(InvalidData):
            self.load([csv_operation(sender_account_id="wrong")])

    def test_malformed_utf8_and_csv(self):
        for content in (b"\xff", b'transaction_id,timestamp\n"unterminated'):
            with self.subTest(content=content), self.assertRaises(InvalidData):
                load_dataset(CLIENTS, content, content)


class ProductTests(unittest.TestCase):
    def model(self):
        model = Mock()
        model.available_from = datetime(2026, 10, 1)
        model.predict.side_effect = lambda rows: [Prediction(0.25, -1.0986122887, ()) for row in rows]
        return model

    def test_missing_fields_and_new_client_produce_finite_inputs_and_warnings(self):
        data = Dataset((), (), (sample_operation(device_id=None, region=None, city=None, balance_before_rub=None),))
        analysis = AnalyzeOperations(CausalFeatureEngine(), self.model()).execute(data)
        self.assertEqual(analysis.assessments[0].prior_count, 0)
        self.assertGreaterEqual(len(analysis.assessments[0].warnings), 5)

    def test_history_detail_excludes_current_same_time_and_future(self):
        current = sample_operation()
        prior = sample_operation("prior", timestamp=datetime(2026, 10, 15))
        future = sample_operation("future", timestamp=datetime(2026, 10, 17))
        same = sample_operation("same")
        data = replace(sample_dataset(), history=(future, prior, same))
        analysis = AnalyzeOperations(CausalFeatureEngine(), self.model()).execute(data)
        self.assertEqual(analysis.assessments[0].prior_count, 1)
        self.assertEqual([op.transaction_id for op in analysis.prior_operations(analysis.assessments[0])], ["prior"])

    def test_empty_and_pre_model_operations_are_rejected(self):
        service = AnalyzeOperations(CausalFeatureEngine(), self.model())
        with self.assertRaises(InvalidData):
            service.execute(replace(sample_dataset(), current=()))
        with self.assertRaises(InvalidData):
            service.execute(replace(sample_dataset(), current=(sample_operation(timestamp=datetime(2026, 9, 30)),)))

    def test_csv_export_preserves_ids_and_threshold_changes_without_rescoring(self):
        model = self.model()
        analysis = AnalyzeOperations(CausalFeatureEngine(), model).execute(sample_dataset())
        for threshold, expected in ((0.2, "1"), (0.3, "0")):
            row = list(csv.DictReader(io.StringIO(export_csv(analysis, threshold).decode("utf-8-sig"))))[0]
            self.assertEqual(row["transaction_id"], "new")
            self.assertEqual(row["review"], expected)
        self.assertEqual(model.predict.call_count, 1)

    def test_nonfinite_threshold_rejected(self):
        for value in (float("nan"), float("inf"), -1, 2):
            with self.subTest(value=value), self.assertRaises(InvalidData):
                valid_threshold(value)
