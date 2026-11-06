"""Causal directed transfer graph and a regularized graph classifier."""

from collections import Counter, defaultdict, deque
from itertools import groupby
from operator import itemgetter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EDGE_FIELDS = (
    "timestamp",
    "sender_account_id",
    "recipient_account_id",
    "amount_rub",
    "transaction_type",
    "status",
)
ONE_HOUR = pd.Timedelta(hours=1)
ONE_DAY = pd.Timedelta(days=1)


class TransferGraph:
    def __init__(self):
        self.outgoing = defaultdict(Counter)
        self.incoming = defaultdict(Counter)
        self.recent_outgoing = defaultdict(deque)
        self.recent_incoming = defaultdict(deque)

    def add(self, sender, recipient, amount, timestamp):
        self.outgoing[sender][recipient] += 1
        self.incoming[recipient][sender] += 1
        self.recent_outgoing[sender].append((timestamp, recipient, amount))
        self.recent_incoming[recipient].append((timestamp, sender, amount))

    @staticmethod
    def recent_summary(events, timestamp, amount):
        start = timestamp - ONE_DAY
        while events and events[0][0] < start:
            events.popleft()
        hour_start = timestamp - ONE_HOUR
        recent = [(account, value) for time, account, value in events if time >= hour_start]
        return (
            len(events),
            len(recent),
            len({account for account, _ in recent}),
            sum(value for _, value in recent),
            sum(abs(value - amount) <= 0.1 * amount for _, value in recent),
        )

    def describe(self, sender, recipient, amount, timestamp):
        sender_out = self.outgoing.get(sender, {})
        sender_in = self.incoming.get(sender, {})
        recipient_out = self.outgoing.get(recipient, {})
        recipient_in = self.incoming.get(recipient, {})
        values = {
            "sender_out_degree": len(sender_out),
            "sender_in_degree": len(sender_in),
            "recipient_out_degree": len(recipient_out),
            "recipient_in_degree": len(recipient_in),
            "direct_edge_count": sender_out.get(recipient, 0),
            "reverse_edge_count": recipient_out.get(sender, 0),
            "two_hop_forward_paths": len(sender_out.keys() & recipient_in.keys()),
            "two_hop_return_paths": len(recipient_out.keys() & sender_in.keys()),
            "shared_senders": len(sender_in.keys() & recipient_in.keys()),
            "shared_recipients": len(sender_out.keys() & recipient_out.keys()),
        }
        directions = {
            "sender_out": self.recent_outgoing.get(sender),
            "sender_in": self.recent_incoming.get(sender),
            "recipient_out": self.recent_outgoing.get(recipient),
            "recipient_in": self.recent_incoming.get(recipient),
        }
        fields = (
            "count_24h",
            "count_1h",
            "unique_accounts_1h",
            "volume_1h",
            "similar_amount_count_1h",
        )
        for direction, events in directions.items():
            summary = self.recent_summary(events, timestamp, amount) if events else (0,) * len(fields)
            values.update({f"{direction}_{field}": value for field, value in zip(fields, summary)})
        return values


def build_graph_features(features):
    """Extract features before adding completed transfers at each timestamp."""
    timestamps = features["timestamp"]
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Graph input must have valid timestamps in chronological order.")

    graph = TransferGraph()
    feature_rows = []
    rows = features.loc[:, list(EDGE_FIELDS)].itertuples(index=False, name=None)
    for timestamp, batch in groupby(rows, key=itemgetter(0)):
        transfers = []
        for _, sender, recipient, amount, kind, status in batch:
            valid_amount = pd.notna(amount) and np.isfinite(amount) and amount > 0
            graph_amount = float(amount) if valid_amount else 0.0
            feature_rows.append(graph.describe(sender, recipient, graph_amount, timestamp))
            if (
                pd.notna(sender)
                and pd.notna(recipient)
                and valid_amount
                and pd.notna(kind)
                and pd.notna(status)
                and kind == "перевод"
                and status == "исполнена"
            ):
                transfers.append((sender, recipient, graph_amount, timestamp))
        for transfer in transfers:
            graph.add(*transfer)

    return pd.DataFrame(feature_rows, index=features.index)


def fit_predict(features, train_mask, validation_mask):
    """Select logistic regularization on validation, then score every row."""
    train_mask = np.asarray(train_mask, dtype=bool)
    validation_mask = np.asarray(validation_mask, dtype=bool)
    graph_features = build_graph_features(features)
    inputs = np.log1p(graph_features.to_numpy(dtype=float))
    train_labels = features.loc[train_mask, "is_fraud"].to_numpy(dtype=int)
    validation_labels = features.loc[validation_mask, "is_fraud"].to_numpy(dtype=int)

    candidates = []
    best_model = None
    best_ap = -1.0
    best_c = None
    for c in (0.03, 0.3, 3.0):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=500))
        model.fit(inputs[train_mask], train_labels)
        scores = model.predict_proba(inputs[validation_mask])[:, 1]
        validation_ap = float(average_precision_score(validation_labels, scores))
        candidates.append({"C": c, "validation_ap": validation_ap})
        if validation_ap > best_ap:
            best_model = model
            best_ap = validation_ap
            best_c = c

    return best_model.predict_proba(inputs)[:, 1], {
        "model": "L2 logistic regression on temporal transfer graph features",
        "parameters": {"C": best_c},
        "feature_count": len(graph_features.columns),
        "candidates": candidates,
    }
