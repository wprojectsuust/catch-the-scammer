"""Use cases depend on the domain and ports implemented by adapters."""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from antifraud.domain import Assessment, Dataset, FeatureRow, InvalidData, Operation, Prediction


class FeatureEngine(Protocol):
    def prepare(self, dataset: Dataset) -> Sequence[FeatureRow]: ...


class RiskModel(Protocol):
    available_from: datetime

    def predict(self, rows: Sequence[FeatureRow]) -> Sequence[Prediction]: ...


@dataclass
class Analysis:
    assessments: tuple[Assessment, ...]
    history_by_account: dict[str, tuple[Operation, ...]]
    history_times: dict[str, tuple[datetime, ...]]
    warnings: tuple[str, ...]

    def prior_operations(self, assessment: Assessment) -> tuple[Operation, ...]:
        account = assessment.operation.sender_account_id
        stop = bisect_left(self.history_times[account], assessment.operation.timestamp)
        return self.history_by_account[account][:stop]


class AnalyzeOperations:
    def __init__(self, features: FeatureEngine, model: RiskModel):
        self.features = features
        self.model = model

    def execute(self, dataset: Dataset) -> Analysis:
        if not dataset.current:
            raise InvalidData("Файл новых операций пуст.")
        if any(op.timestamp < self.model.available_from for op in dataset.current):
            raise InvalidData("Новые операции должны быть позже периода настройки модели.")
        rows = self.features.prepare(dataset)
        predictions = self.model.predict(rows)
        if len(rows) != len(predictions):
            raise RuntimeError("Model returned a different number of predictions.")
        assessments = tuple(
            Assessment(row.operation, prediction, row.warnings, int(row.values["prior_count"]))
            for row, prediction in zip(rows, predictions)
        )
        histories = defaultdict(list)
        for op in sorted((*dataset.history, *dataset.current), key=lambda op: (op.timestamp, op.transaction_id)):
            histories[op.sender_account_id].append(op)
        history_by_account = {account: tuple(items) for account, items in histories.items()}
        return Analysis(
            assessments,
            history_by_account,
            {account: tuple(op.timestamp for op in items) for account, items in history_by_account.items()},
            dataset.warnings,
        )
