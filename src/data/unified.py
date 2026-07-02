from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.amlsim import build_unified_amlsim_df
from src.data.samld import build_unified_samld_df
from src.data.amlworld import build_unified_amlworld_df
from src.data.bitcoin_alpha import build_unified_bitcoin_alpha_df
from src.data.bitcoin_otc import build_unified_bitcoin_otc_df


REQUIRED_UNIFIED_COLS = {"src", "dst", "timestamp", "amount", "label"}


def load_unified_df(
    dataset_name: str,
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_rows: int | None = None,
):
    name = dataset_name.lower()

    if name == "amlsim":
        return build_unified_amlsim_df(
            csv_path=csv_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            max_rows=max_rows,
        )

    if name in {"samld", "saml-d"}:
        return build_unified_samld_df(
            csv_path=csv_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            max_rows=max_rows,
        )

    if name == "amlworld_hi_small":
        return build_unified_amlworld_df(
            csv_path=csv_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            max_rows=max_rows,
        )

    if name in {"bitcoin_alpha", "btc_alpha", "alpha"}:
        return build_unified_bitcoin_alpha_df(
            csv_path=csv_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            max_rows=max_rows,
        )

    if name in {"bitcoin_otc", "btc_otc", "otc"}:
        return build_unified_bitcoin_otc_df(
            csv_path=csv_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            max_rows=max_rows,
        )

    raise ValueError(f"Unknown dataset_name={dataset_name}")


def _make_node_features(df: pd.DataFrame, num_nodes: int, mode: str) -> torch.Tensor:
    if mode == "constant":
        return torch.ones((num_nodes, 1), dtype=torch.float)

    if mode == "degree":
        in_deg = (
            df.groupby("dst")
            .size()
            .reindex(range(num_nodes), fill_value=0)
            .to_numpy(dtype=np.float32)
        )
        out_deg = (
            df.groupby("src")
            .size()
            .reindex(range(num_nodes), fill_value=0)
            .to_numpy(dtype=np.float32)
        )
        x_np = np.stack(
            [np.log1p(np.maximum(in_deg + out_deg, 0.0))],
            axis=1,
        ).astype(np.float32)
        return torch.tensor(x_np, dtype=torch.float)

    raise ValueError(f"Unknown node feature mode: {mode}")


def _collect_numeric_edge_feature_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in {"src", "dst", "timestamp", "amount", "label"}:
            continue
        if df[c].dtype.kind in {"i", "u", "f", "b"}:
            cols.append(c)
    return cols


def build_transaction_graph_no_edge_features(
    df: pd.DataFrame,
    num_nodes: int,
    node_feature_mode: str = "constant",
) -> Data:
    """
    Used by non-FraudGT models.
    Does not expose raw transaction edge features.
    """
    missing = REQUIRED_UNIFIED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Unified dataframe missing required columns: {sorted(missing)}")

    src = torch.tensor(df["src"].to_numpy(), dtype=torch.long)
    dst = torch.tensor(df["dst"].to_numpy(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    edge_time = torch.tensor(
        df["timestamp"].to_numpy(dtype=np.float32),
        dtype=torch.float,
    )
    edge_label = torch.tensor(
        df["label"].to_numpy(dtype=np.float32),
        dtype=torch.float,
    )

    x = _make_node_features(df, num_nodes, node_feature_mode)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_time=edge_time,
        edge_label=edge_label,
        num_nodes=num_nodes,
    )


def build_transaction_graph_with_edge_features(
    df: pd.DataFrame,
    num_nodes: int,
    node_feature_mode: str = "constant",
) -> Data:
    """
    Used by FraudGT family.
    Exposes raw transaction edge attributes.
    """
    missing = REQUIRED_UNIFIED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Unified dataframe missing required columns: {sorted(missing)}")

    src = torch.tensor(df["src"].to_numpy(), dtype=torch.long)
    dst = torch.tensor(df["dst"].to_numpy(), dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    edge_time = torch.tensor(
        df["timestamp"].to_numpy(dtype=np.float32),
        dtype=torch.float,
    )
    edge_label = torch.tensor(
        df["label"].to_numpy(dtype=np.float32),
        dtype=torch.float,
    )

    edge_feature_cols = ["amount"] + _collect_numeric_edge_feature_cols(df)

    if len(edge_feature_cols) == 0:
        edge_attr = torch.zeros((len(df), 1), dtype=torch.float)
    else:
        edge_attr = torch.tensor(
            df[edge_feature_cols].to_numpy(dtype=np.float32),
            dtype=torch.float,
        )

    x = _make_node_features(df, num_nodes, node_feature_mode)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_time=edge_time,
        edge_label=edge_label,
        num_nodes=num_nodes,
    )