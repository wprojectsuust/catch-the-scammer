"""Shared serialization and export for CLI and browser."""

import csv
import io
import json
from dataclasses import asdict

from antifraud.domain import InvalidData


FEATURE_LABELS = {
    "amount_log": "Сумма операции", "balance_fraction": "Доля суммы в остатке",
    "hour_sin": "Время суток (sin)", "hour_cos": "Время суток (cos)",
    "is_new_device": "Новое устройство", "is_new_recipient": "Новый получатель",
    "account_age_days": "Возраст счёта", "income_log": "Доход клиента",
    "prior_count": "Число прошлых операций", "amount_robust_z": "Отклонение суммы от обычной",
    "velocity_5m": "Операции за 5 минут", "velocity_60m": "Операции за час",
    "velocity_24h": "Операции за сутки", "region_seen": "Регион встречался ранее",
    "device_seen": "Устройство встречалось ранее", "recipient_seen": "Получатель встречался ранее",
    "geo_change": "Смена региона", "transaction_type": "Тип операции", "region": "Регион",
    "city": "Город", "account_type": "Тип счёта", "client_category": "Категория клиента",
}


def valid_threshold(value):
    if not 0 <= value <= 1:
        raise InvalidData("Порог должен быть конечным числом от 0 до 1.")
    return value


def operation_dict(operation):
    result = asdict(operation)
    result["timestamp"] = operation.timestamp.isoformat(sep=" ")
    return result


def assessment_dict(assessment, threshold):
    prediction = assessment.prediction
    contributions = sorted(prediction.contributions, key=lambda item: abs(item.impact), reverse=True)
    return {
        **operation_dict(assessment.operation),
        "score": prediction.score,
        "review": prediction.score >= threshold,
        "category": "review" if prediction.score >= threshold else "routine",
        "prior_count": assessment.prior_count,
        "warnings": assessment.warnings,
        "base_value": prediction.base_value,
        "contributions": [{**asdict(item), "label": FEATURE_LABELS.get(item.feature, item.feature)}
                          for item in contributions],
    }


def export_csv(analysis, threshold):
    valid_threshold(threshold)
    target = io.StringIO(newline="")
    fields = ["transaction_id", "timestamp", "risk_score", "category", "review", "explanation", "limitations"]
    writer = csv.DictWriter(target, fields)
    writer.writeheader()
    for item in sorted(analysis.assessments, key=lambda item: item.prediction.score, reverse=True):
        detail = assessment_dict(item, threshold)
        # Protect spreadsheet formula prefixes while preserving ordinary identifiers.
        writer.writerow({
            "transaction_id": ("'" + item.operation.transaction_id
                               if item.operation.transaction_id.lstrip().startswith(("=", "+", "-", "@"))
                               else item.operation.transaction_id),
            "timestamp": item.operation.timestamp.isoformat(),
            "risk_score": f"{item.prediction.score:.8f}",
            "category": detail["category"],
            "review": int(detail["review"]),
            "explanation": json.dumps(detail["contributions"][:5], ensure_ascii=False),
            "limitations": json.dumps(item.warnings, ensure_ascii=False),
        })
    return target.getvalue().encode("utf-8-sig")
