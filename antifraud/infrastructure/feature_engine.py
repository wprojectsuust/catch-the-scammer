from dataclasses import asdict

import pandas as pd

from antifraud.domain import Dataset, FeatureRow
from antifraud.infrastructure.temporal import build_features
from antifraud.infrastructure.tabular_model import CATEGORICAL, NUMERIC


class CausalFeatureEngine:
    def prepare(self, dataset: Dataset):
        operations = sorted((*dataset.history, *dataset.current), key=lambda op: (op.timestamp, op.transaction_id))
        clients = {client.client_id: client for client in dataset.clients}
        current_ids = {op.transaction_id for op in dataset.current}
        records = []
        for operation in operations:
            record = asdict(operation)
            client = clients.get(operation.customer_id)
            record.update({
                "registration_date": client.registration_date if client else None,
                "account_type": client.account_type if client else None,
                "client_category": client.client_category if client else None,
                "monthly_income_rub": client.monthly_income_rub if client else None,
            })
            records.append(record)
        frame = pd.DataFrame(records)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["registration_date"] = pd.to_datetime(frame["registration_date"])
        frame = build_features(frame, use_supplied_flags=False)
        rows = []
        for operation, values in zip(operations, frame.to_dict("records")):
            if operation.transaction_id not in current_ids:
                continue
            warnings = []
            if operation.customer_id not in clients:
                warnings.append("Клиент отсутствует в справочнике.")
            else:
                client = clients[operation.customer_id]
                if client.monthly_income_rub is None:
                    warnings.append("Доход клиента отсутствует.")
                if client.registration_date is None:
                    warnings.append("Дата регистрации клиента отсутствует.")
            if values["prior_count"] < 5:
                warnings.append("Меньше пяти прошлых операций — индивидуальная оценка суммы ограничена.")
            if operation.balance_before_rub is None or operation.balance_before_rub == 0:
                warnings.append("Доля списания в остатке недоступна.")
            for field, label in (("device_id", "устройство"), ("region", "регион"), ("city", "город")):
                if getattr(operation, field) is None:
                    warnings.append(f"Отсутствует {label}.")
            predictors = {key: None if pd.isna(values[key]) else values[key] for key in (*NUMERIC, *CATEGORICAL)}
            rows.append(FeatureRow(operation, predictors, tuple(warnings)))
        return rows
