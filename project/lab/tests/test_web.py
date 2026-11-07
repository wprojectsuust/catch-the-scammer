import csv
import io
import unittest

from fastapi.testclient import TestClient

from antifraud.interfaces.web import create_app
from tests import test_product
from tests.test_product import CLIENTS, FIELDS, csv_bytes, csv_operation


class WebTests(unittest.TestCase):
    def setUp(self):
        self.model = test_product.ProductTests().model()
        self.model.default_threshold = 0.2
        self.client = TestClient(create_app(self.model))

    def upload(self, current):
        return self.client.post("/api/analyses", files={
            "clients": ("clients.csv", CLIENTS),
            "history": ("history.csv", csv_bytes([csv_operation("prior", timestamp="2026-10-15")], FIELDS)),
            "current": ("new.csv", current),
        })

    def test_upload_queue_replay_details_and_export_share_one_scoring_run(self):
        response = self.upload(csv_bytes([
            csv_operation("first"), csv_operation("second", timestamp="2026-10-16 12:01:00"),
        ], FIELDS))
        self.assertEqual(response.status_code, 200, response.text)
        base = f"/api/analyses/{response.json()['id']}"
        replay = self.client.get(base, params={"threshold": 0.2, "seen": 1}).json()
        self.assertEqual((replay["total"], replay["observed"], replay["review_count"]), (2, 1, 1))
        self.assertEqual(replay["rows"][0]["transaction_id"], "first")
        detail = self.client.get(base + "/operations/second", params={"threshold": 0.2}).json()
        self.assertEqual([row["transaction_id"] for row in detail["history"]], ["prior", "first"])
        filtered = self.client.get(base, params={"threshold": 0.3, "review_only": True}).json()
        self.assertEqual(filtered["rows"], [])
        export = self.client.get(base + "/export", params={"threshold": 0.3})
        rows = list(csv.DictReader(io.StringIO(export.content.decode("utf-8-sig"))))
        self.assertEqual({row["transaction_id"]: row["review"] for row in rows}, {"first": "0", "second": "0"})
        self.assertEqual(self.model.predict.call_count, 1)

    def test_sessions_are_isolated_and_bad_input_is_reported(self):
        first = self.upload(csv_bytes([csv_operation("first")], FIELDS)).json()["id"]
        second = self.upload(csv_bytes([csv_operation("second")], FIELDS)).json()["id"]
        self.assertNotEqual(first, second)
        missing = self.client.get(f"/api/analyses/{first}/operations/second", params={"threshold": 0.2})
        self.assertEqual(missing.status_code, 404)
        invalid = self.upload(b"malformed\n")
        self.assertEqual(invalid.status_code, 422)
        self.assertIn("отсутствуют столбцы", invalid.json()["detail"])

    def test_browser_assets_and_scenario_input(self):
        self.assertIn("Очередь проверок", self.client.get("/").text)
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(len(self.client.get("/api/scenarios").json()), 7)
        response = self.client.post("/api/scenarios/ordinary", json={"amount": 2500})
        self.assertEqual(response.status_code, 200, response.text)
        base = f"/api/analyses/{response.json()['id']}"
        self.assertEqual(self.client.get(base, params={"threshold": 0.2}).json()["rows"][0]["amount_rub"], 2500)
        self.assertEqual(self.client.post("/api/scenarios/ordinary", json={"amount": -1}).status_code, 422)
