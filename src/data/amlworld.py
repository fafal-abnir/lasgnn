from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.splits import temporal_split_indices


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
    return df


def build_unified_amlworld_df(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, int, int, int]:
    """
    AMLWorld-HI-Small parser.

    Expected important raw columns:
        Account
        Account.1
        Amount Received
        Is Laundering
        Timestamp

    Optional edge categorical columns:
        From Bank
        To Bank
        Receiving Currency
        Payment Currency
        Payment Format

    Unified output:
        src, dst, timestamp, amount, label, from_bank, to_bank,
        receiving_currency, payment_currency, payment_format
    """

    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)

    rename_map = {
        "account": "src",
        "account.1": "dst",
        "amount_received": "amount",
        "is_laundering": "label",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["src", "dst", "amount", "label", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"AMLWorld missing required columns: {missing}. Found: {list(df.columns)}"
        )

    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    df = df.dropna(subset=required).copy()

    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    df["date_time"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    if df["date_time"].isna().all():
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["date_time"] = pd.to_datetime(
            df["timestamp"],
            unit="s",
            origin="unix",
            errors="coerce",
            utc=True,
        )
    else:
        df["timestamp"] = df["date_time"].astype("int64") / 1e9

    df = df.dropna(subset=["amount", "timestamp", "date_time"]).copy()
    df = df.sort_values("date_time").reset_index(drop=True)

    n = len(df)
    train_end, val_end = temporal_split_indices(
        n,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    all_ids = pd.unique(df[["src", "dst"]].values.ravel())
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}

    df["src"] = df["src"].map(id_mapping).astype(np.int64)
    df["dst"] = df["dst"].map(id_mapping).astype(np.int64)

    min_ts = df["timestamp"].min()
    df["timestamp"] = (df["timestamp"] - min_ts).astype(np.float64)

    df["amount"] = np.log1p(df["amount"].clip(lower=0))

    scaler = StandardScaler()
    if train_end > 0:
        df.loc[: train_end - 1, ["amount"]] = scaler.fit_transform(
            df.loc[: train_end - 1, ["amount"]]
        )
    if train_end < len(df):
        df.loc[train_end:, ["amount"]] = scaler.transform(
            df.loc[train_end:, ["amount"]]
        )

    cat_cols = [
        "from_bank",
        "to_bank",
        "receiving_currency",
        "payment_currency",
        "payment_format",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]

    if cat_cols:
        df[cat_cols] = df[cat_cols].astype(str).fillna("__nan__")

        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        if train_end > 0:
            enc.fit(df.loc[: train_end - 1, cat_cols])
            df.loc[:, cat_cols] = enc.transform(df[cat_cols])
        else:
            df.loc[:, cat_cols] = 0.0

        for c in cat_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.float32)

    ordered_cols = ["src", "dst", "timestamp", "amount", "label"] + cat_cols
    df = df[ordered_cols].copy()

    num_nodes = int(max(df["src"].max(), df["dst"].max()) + 1)

    print(
        f"[AMLWorld] nodes={num_nodes}, edges={len(df)}, "
        f"positive_rate={df['label'].mean():.6f}, edge_features={['amount'] + cat_cols}"
    )

    return df, num_nodes, train_end, val_end