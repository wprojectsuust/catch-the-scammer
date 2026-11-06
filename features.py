from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby

import numpy as np
import pandas as pd


@dataclass
class AccountHistory:
    times: list = field(default_factory=list)
    log_amounts: list = field(default_factory=list)
    devices: set = field(default_factory=set)
    recipients: set = field(default_factory=set)
    regions: set = field(default_factory=set)
    last_region: str = ""


def load_inputs(data_dir):
    operations = pd.read_csv(data_dir / "transactions_history.csv", parse_dates=["timestamp"])
    clients = pd.read_csv(data_dir / "clients.csv", parse_dates=["registration_date"])
    if operations["transaction_id"].isna().any() or operations["transaction_id"].duplicated().any():
        raise ValueError("Transaction identifiers must be present and unique.")
    if operations["timestamp"].isna().any():
        raise ValueError("A valid timestamp is required for each transaction.")
    if not operations["is_fraud"].dropna().isin([0, 1]).all():
        raise ValueError("Expected fraud labels 0, 1, or missing.")
    client_columns = ["client_id", "registration_date", "account_type", "client_category", "monthly_income_rub"]
    return operations.merge(
        clients[client_columns], left_on="customer_id", right_on="client_id", how="left", validate="many_to_one"
    ).sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def build_features(operations, *, use_supplied_flags=True):
    features = operations.copy()
    features["amount_log"] = np.log1p(features["amount_rub"].clip(lower=0))
    features["balance_fraction"] = features["amount_rub"].div(features["balance_before_rub"].where(
        features["balance_before_rub"] > 0
    ))
    hours = features["timestamp"].dt.hour + features["timestamp"].dt.minute / 60
    features["hour_sin"] = np.sin(hours * np.pi / 12)
    features["hour_cos"] = np.cos(hours * np.pi / 12)
    features["account_age_days"] = (features["timestamp"] - features["registration_date"]).dt.days
    features["income_log"] = np.log1p(features["monthly_income_rub"].clip(lower=0))

    accounts = defaultdict(AccountHistory)
    records = []
    columns = ["timestamp", "sender_account_id", "recipient_account_id", "device_id", "region", "amount_log"]
    for timestamp, group in groupby(features[columns].fillna({"device_id": "", "region": ""}).itertuples(
        index=False, name=None
    ), key=lambda row: row[0]):
        rows = list(group)
        for _, sender, recipient, device, region, amount in rows:
            history = accounts[sender]
            robust_z = 0.0
            if len(history.log_amounts) >= 5 and pd.notna(amount):
                amounts = np.asarray(history.log_amounts)
                median = np.median(amounts)
                mad = np.median(np.abs(amounts - median))
                robust_z = (amount - median) / max(1.4826 * mad, 0.25)
            now = timestamp.timestamp()
            # The time list contains events strictly before this timestamp.
            windows = [len(history.times) - np.searchsorted(history.times, now - seconds)
                       for seconds in (300, 3600, 86400)]
            records.append((len(history.times), robust_z, *windows, int(region in history.regions),
                            int(device in history.devices), int(recipient in history.recipients),
                            int(bool(region and history.last_region and region != history.last_region))))
        for _, sender, recipient, device, region, amount in rows:
            history = accounts[sender]
            history.times.append(timestamp.timestamp())
            if pd.notna(amount):
                history.log_amounts.append(amount)
            if device:
                history.devices.add(device)
            if region:
                history.regions.add(region)
                history.last_region = region
            if pd.notna(recipient):
                history.recipients.add(recipient)

    causal_columns = ["prior_count", "amount_robust_z", "velocity_5m", "velocity_60m", "velocity_24h",
                      "region_seen", "device_seen", "recipient_seen", "geo_change"]
    features[causal_columns] = pd.DataFrame(records, index=features.index, columns=causal_columns)
    if not use_supplied_flags:
        features["is_new_device"] = (1 - features["device_seen"]).where(features["device_id"].notna())
        features["is_new_recipient"] = (1 - features["recipient_seen"]).where(features["recipient_account_id"].notna())
    return features
