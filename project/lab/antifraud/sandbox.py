"""Synthetic scenarios exercise the production use case without changing the model."""

import json
import math
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from antifraud.application import AnalyzeOperations
from antifraud.domain import Client, Dataset, InvalidData, Operation
from antifraud.infrastructure.feature_engine import CausalFeatureEngine
from antifraud.interfaces.presentation import assessment_dict


SCENARIOS = {
    "ordinary": ("Обычная операция", "Привычная сумма, устройство и получатель."),
    "large_purchase": ("Крупная законная покупка", "Синтетическая законная покупка. Проверяем реакцию на аномальную сумму."),
    "burst": ("Серия быстрых переводов", "Три перевода за предыдущие три минуты."),
    "new_device": ("Новое устройство и получатель", "Два признака новизны появляются одновременно."),
    "geography": ("Смена региона", "Географический контекст ограничен регионом и городом."),
    "new_client": ("Новый клиент", "Оценка без истории операций."),
    "missing": ("Неполные данные", "Отсутствуют устройство, география и остаток."),
}


def scenario_dataset(name, available_from, amount=None):
    if name not in SCENARIOS:
        raise InvalidData("Неизвестный сценарий.")
    now = available_from + timedelta(days=40, hours=12)
    client = Client("DEMO-C", "DEMO-A", now - timedelta(days=400), "дебетовый", "наёмный сотрудник", 65000)
    operation = Operation(
        "DEMO-CURRENT", "DEMO-A", "DEMO-B", "DEMO-C", now, 1200, 75000,
        "перевод", "DEMO-DEVICE", "Башкортостан", "Уфа", "исполнена",
    )
    history = tuple(replace(operation, transaction_id=f"DEMO-{i:03d}",
                            timestamp=now - timedelta(days=30 - i), amount_rub=1000 + (i % 5) * 100)
                    for i in range(30))
    if name == "large_purchase":
        operation = replace(operation, amount_rub=50000, transaction_type="покупка")
    elif name == "burst":
        history += tuple(replace(operation, transaction_id=f"BURST-{i}",
                                 timestamp=now - timedelta(minutes=i), recipient_account_id=f"BURST-R{i}")
                         for i in (3, 2, 1))
    elif name == "new_device":
        operation = replace(operation, device_id="NEW-DEVICE", recipient_account_id="NEW-RECIPIENT")
    elif name == "geography":
        operation = replace(operation, region="Москва", city="Москва")
    elif name == "new_client":
        history = ()
        client = replace(client, registration_date=now)
    elif name == "missing":
        operation = replace(operation, device_id=None, region=None, city=None, balance_before_rub=None)
    if amount is not None:
        if not math.isfinite(amount) or amount <= 0:
            raise InvalidData("Сумма сценария должна быть конечным положительным числом.")
        operation = replace(operation, amount_rub=amount)
    return Dataset((client,), history, (operation,), ("Синтетический сценарий. Метрики качества на нём не измеряются.",))


def verify_scenario(name, row, assessment):
    values = row.values
    checks = {
        "finite_score": math.isfinite(assessment.prediction.score) and 0 <= assessment.prediction.score <= 1,
        "causal_history": assessment.prior_count == (0 if name == "new_client" else 33 if name == "burst" else 30),
    }
    raw_score = assessment.prediction.base_value + sum(item.impact for item in assessment.prediction.contributions)
    explained_score = 1 / (1 + math.exp(-raw_score))
    checks["explanation_additivity"] = abs(explained_score - assessment.prediction.score) < 1e-8
    if name == "ordinary":
        checks["familiar_device"] = values["is_new_device"] == 0
    elif name == "large_purchase":
        checks["unusual_amount"] = values["amount_robust_z"] > 4
    elif name == "burst":
        checks["velocity"] = values["velocity_5m"] == 3
    elif name == "new_device":
        checks["novelty"] = values["is_new_device"] == values["is_new_recipient"] == 1
    elif name == "geography":
        checks["geography"] = values["geo_change"] == 1 and values["region_seen"] == 0
    elif name in ("new_client", "missing"):
        checks["limitations_visible"] = bool(assessment.warnings)
    return checks


def run_scenarios(model):
    engine = CausalFeatureEngine()
    service = AnalyzeOperations(engine, model)
    results = []
    for name, (title, description) in SCENARIOS.items():
        dataset = scenario_dataset(name, model.available_from)
        assessment = service.execute(dataset).assessments[0]
        checks = verify_scenario(name, engine.prepare(dataset)[0], assessment)
        results.append({
            "scenario": name, "title": title, "description": description,
            "passed": all(checks.values()), "checks": checks,
            "assessment": assessment_dict(assessment, model.default_threshold),
        })
    return {
        "purpose": "Synthetic mechanism checks; not a labelled quality benchmark.",
        "model_sha256": model.metadata["model_sha256"],
        "passed": all(result["passed"] for result in results),
        "scenarios": results,
    }


def run_and_save(model, output):
    result = run_scenarios(model)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise InvalidData("Проверка песочницы выявила нарушение. См. отчёт.")
    print(f"Пройдено сценариев: {len(result['scenarios'])}. Отчёт: {output}")
