"""Strict core schema with tolerant optional fields and explicit diagnostics."""

import csv
import io
import math
from datetime import datetime

from antifraud.domain import Client, Dataset, InvalidData, Operation


MAX_ROWS = 100_000
OP_REQUIRED = {"transaction_id", "sender_account_id", "recipient_account_id", "customer_id",
               "timestamp", "amount_rub", "transaction_type", "status", "currency"}


def records(content: bytes, required: set[str], name: str):
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")), strict=True)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise InvalidData(f"{name}: отсутствуют столбцы {', '.join(sorted(missing))}.")
        for number, record in enumerate(reader, start=2):
            if number > MAX_ROWS + 1:
                raise InvalidData(f"{name}: превышен лимит {MAX_ROWS} строк.")
            if None in record or any(value is None for value in record.values()):
                raise InvalidData(f"{name}, строка {number}: неверное число полей.")
            yield number, {key: value.strip() for key, value in record.items()}
    except (UnicodeError, csv.Error) as exc:
        raise InvalidData(f"{name}: требуется корректный CSV в UTF-8.") from exc


def optional_number(value, context, warnings):
    if value in (None, ""):
        return None
    try:
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError()
        return number
    except ValueError:
        warnings.append(f"{context}: некорректное числовое поле обработано как пропуск.")
        return None


def parse_time(value, context):
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is not None:
            raise ValueError()
        if not 1900 <= result.year <= 2100:
            raise ValueError()
        return result
    except (TypeError, ValueError):
        raise InvalidData(f"{context}: ожидается дата без часового пояса, например 2026-10-16 12:00:00.") from None


def unique_rows(rows, key, name, warnings):
    result = {}
    duplicates = 0
    for row in rows:
        identity = getattr(row, key)
        if identity in result:
            if row != result[identity]:
                raise InvalidData(f"{name}: конфликтующие записи с ID {identity}.")
            duplicates += 1
        result[identity] = row
    if duplicates:
        warnings.append(f"{name}: удалено точных повторов — {duplicates}.")
    return tuple(result.values())


def load_clients(content, warnings):
    clients = []
    for number, row in records(content, {"client_id", "account_id"}, "clients.csv"):
        context = f"clients.csv, строка {number}"
        if not row["client_id"] or not row["account_id"]:
            raise InvalidData(f"{context}: отсутствует ID клиента или счёта.")
        registered = None
        if row.get("registration_date"):
            registered = parse_time(row["registration_date"], context)
        clients.append(Client(
            row["client_id"], row["account_id"], registered,
            row.get("account_type") or None, row.get("client_category") or None,
            optional_number(row.get("monthly_income_rub"), context, warnings),
        ))
    clients = unique_rows(clients, "client_id", "clients.csv", warnings)
    if len({client.account_id for client in clients}) != len(clients):
        raise InvalidData("В clients.csv один счёт указан у нескольких клиентов.")
    return clients


def load_operations(content, name, warnings):
    operations = []
    for number, row in records(content, OP_REQUIRED, name):
        context = f"{name}, строка {number}"
        if any(not row[key] for key in OP_REQUIRED):
            raise InvalidData(f"{context}: пропущено обязательное поле.")
        amount = optional_number(row["amount_rub"], context, warnings)
        if amount is None or amount <= 0:
            raise InvalidData(f"{context}: сумма должна быть конечным положительным числом.")
        if row["currency"] != "RUB":
            raise InvalidData(f"{context}: модель поддерживает RUB.")
        if row["status"] not in {"исполнена", "отклонена"}:
            raise InvalidData(f"{context}: неизвестный статус операции.")
        operations.append(Operation(
            row["transaction_id"], row["sender_account_id"], row["recipient_account_id"],
            row["customer_id"], parse_time(row["timestamp"], context), amount,
            optional_number(row.get("balance_before_rub"), context, warnings),
            row["transaction_type"], row.get("device_id") or None, row.get("region") or None,
            row.get("city") or None, row["status"], row["currency"],
        ))
    return unique_rows(operations, "transaction_id", name, warnings)


def load_dataset(clients_csv: bytes, history_csv: bytes, current_csv: bytes) -> Dataset:
    warnings = []
    clients = load_clients(clients_csv, warnings)
    history = load_operations(history_csv, "transactions_history.csv", warnings)
    current = load_operations(current_csv, "transactions_test.csv", warnings)
    if {op.transaction_id for op in history} & {op.transaction_id for op in current}:
        raise InvalidData("ID новых операций пересекаются с историей. Удалите пересечение перед анализом.")
    client_accounts = {client.client_id: client.account_id for client in clients}
    for operation in (*history, *current):
        account = client_accounts.get(operation.customer_id)
        if account is not None and account != operation.sender_account_id:
            raise InvalidData(f"{operation.transaction_id}: отправитель не соответствует клиенту.")
    return Dataset(clients, history, current, tuple(warnings))
