from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.splits import temporal_split_indices


def build_unified_bitcoin_alpha_df(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, int, int, int]:
    """
    Bitcoin Alpha signed network.

    Expected raw columns:
        SOURCE, TARGET, RATING, TIME

    Unified output columns:
        src, dst, timestamp, amount, label
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["source", "target", "rating", "time"],
    )
    df.columns = df.columns.str.lower().str.strip()

    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    required = ["source", "target", "rating", "time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Bitcoin Alpha missing required columns: {missing}. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=required).copy()

    df["source"] = pd.to_numeric(df["source"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    df = df.dropna(subset=["source", "target", "rating", "time"]).copy()

    df["src"] = df["source"].astype(np.int64)
    df["dst"] = df["target"].astype(np.int64)

    # Binary label: negative rating is positive class.
    df["label"] = (df["rating"] < 0).astype(np.int64)

    # Keep timestamp only; do not use rating as feature.
    df["timestamp"] = df["time"].astype(np.float64)

    # Dummy amount so unified schema stays the same.
    df["amount"] = np.ones(len(df), dtype=np.float32)

    df = df.sort_values("timestamp").reset_index(drop=True)

    n = len(df)
    train_end, val_end = temporal_split_indices(n, train_ratio, val_ratio)

    all_ids = pd.unique(df[["src", "dst"]].values.ravel())
    id_map = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
    df["src"] = df["src"].map(id_map).astype(np.int64)
    df["dst"] = df["dst"].map(id_map).astype(np.int64)

    min_ts = df["timestamp"].min()
    df["timestamp"] = (df["timestamp"] - min_ts).astype(np.float64)

    ordered_cols = ["src", "dst", "timestamp", "amount", "label"]
    df = df[ordered_cols].copy()

    num_nodes = int(max(df["src"].max(), df["dst"].max()) + 1)
    return df, num_nodes, train_end, val_end