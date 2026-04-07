from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.splits import temporal_split_indices


def build_unified_samld_df(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, int, int, int]:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    rename_map = {
        "sender_account": "src",
        "receiver_account": "dst",
        "transaction_amount": "amount",
        "amt": "amount",
        "tx_amount": "amount",
        "is_laundering": "label",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required_base = ["src", "dst", "amount", "label"]
    missing_base = [c for c in required_base if c not in df.columns]
    if missing_base:
        raise ValueError(f"SAML-D missing required columns: {missing_base}. Found: {list(df.columns)}")

    if "date" in df.columns:
        time_col = "date"
    elif "time" in df.columns:
        time_col = "time"
    elif "timestamp" in df.columns:
        time_col = "timestamp"
    else:
        raise ValueError("SAML-D missing a time column. Expected one of: ['timestamp', 'time', 'date']")

    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    df = df.dropna(subset=["src", "dst", "amount", "label", time_col]).copy()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    if np.issubdtype(df[time_col].dtype, np.number):
        df["timestamp"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df["timestamp"] = ts.astype("int64") / 1e9

    df = df.dropna(subset=["amount", "timestamp"]).copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end, val_end = temporal_split_indices(n, train_ratio, val_ratio)

    all_ids = pd.unique(df[["src", "dst"]].values.ravel())
    id_map = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
    df["src"] = df["src"].map(id_map).astype(np.int64)
    df["dst"] = df["dst"].map(id_map).astype(np.int64)

    min_ts = df["timestamp"].min()
    df["timestamp"] = (df["timestamp"] - min_ts).astype(np.float64)

    df["amount"] = np.log1p(df["amount"].clip(lower=0))
    scaler = StandardScaler()
    if train_end > 0:
        df.loc[: train_end - 1, ["amount"]] = scaler.fit_transform(df.loc[: train_end - 1, ["amount"]])
    if train_end < len(df):
        df.loc[train_end:, ["amount"]] = scaler.transform(df.loc[train_end:, ["amount"]])

    extra_cols: list[str] = []

    if "payment_currency" in df.columns:
        df["payment_currency"] = df["payment_currency"].astype(str).fillna("__nan__")
        df = pd.get_dummies(df, columns=["payment_currency"], dtype=np.float32)
        extra_cols.extend([c for c in df.columns if c.startswith("payment_currency_")])

    if "received_currency" in df.columns:
        df["received_currency"] = df["received_currency"].astype(str).fillna("__nan__")
        df = pd.get_dummies(df, columns=["received_currency"], dtype=np.float32)
        extra_cols.extend([c for c in df.columns if c.startswith("received_currency_")])

    if "sender_bank_location" in df.columns:
        df["sender_bank_location"] = df["sender_bank_location"].astype(str).fillna("__nan__")
        df = pd.get_dummies(df, columns=["sender_bank_location"], dtype=np.float32)
        extra_cols.extend([c for c in df.columns if c.startswith("sender_bank_location_")])

    if "receiver_bank_location" in df.columns:
        df["receiver_bank_location"] = df["receiver_bank_location"].astype(str).fillna("__nan__")
        df = pd.get_dummies(df, columns=["receiver_bank_location"], dtype=np.float32)
        extra_cols.extend([c for c in df.columns if c.startswith("receiver_bank_location_")])

    if "payment_type" in df.columns:
        df["payment_type"] = df["payment_type"].astype(str).fillna("__nan__")
        df = pd.get_dummies(df, columns=["payment_type"], dtype=np.float32)
        extra_cols.extend([c for c in df.columns if c.startswith("payment_type_")])

    ordered_cols = ["src", "dst", "timestamp", "amount", "label"] + extra_cols
    df = df[ordered_cols].copy()

    num_nodes = int(max(df["src"].max(), df["dst"].max()) + 1)
    return df, num_nodes, train_end, val_end