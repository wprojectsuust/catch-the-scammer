"""Business types. This module has no dependency on data or UI frameworks."""

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping


class InvalidData(ValueError):
    """A user-correctable problem with an input dataset."""


@dataclass(frozen=True)
class Client:
    client_id: str
    account_id: str
    registration_date: datetime | None
    account_type: str | None
    client_category: str | None
    monthly_income_rub: float | None


@dataclass(frozen=True)
class Operation:
    transaction_id: str
    sender_account_id: str
    recipient_account_id: str
    customer_id: str
    timestamp: datetime
    amount_rub: float
    balance_before_rub: float | None
    transaction_type: str
    device_id: str | None
    region: str | None
    city: str | None
    status: str
    currency: str = "RUB"


@dataclass(frozen=True)
class Dataset:
    clients: tuple[Client, ...]
    history: tuple[Operation, ...]
    current: tuple[Operation, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeatureRow:
    operation: Operation
    values: Mapping[str, float | str | None]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Contribution:
    feature: str
    value: float | str | None
    impact: float


@dataclass(frozen=True)
class Prediction:
    score: float
    base_value: float
    contributions: tuple[Contribution, ...]


@dataclass(frozen=True)
class Assessment:
    operation: Operation
    prediction: Prediction
    warnings: tuple[str, ...]
    prior_count: int
