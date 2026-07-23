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


def _prepare_hack_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    After lowercasing:
        timeStamp        -> timestamp
        blockNumber      -> blocknumber
        tokenSymbol      -> tokensymbol
        contractAddress  -> contractaddress
        isError          -> iserror
        gasPrice         -> gasprice
        gasUsed          -> gasused
    """
    rename_map = {
        "from": "src",
        "to": "dst",
        "value": "amount",
        "timestamp": "timestamp",
        "time_stamp": "timestamp",
        "blocknumber": "block_number",
        "block_number": "block_number",
        "tokensymbol": "token_symbol",
        "token_symbol": "token_symbol",
        "contractaddress": "contract_address",
        "contract_address": "contract_address",
        "iserror": "is_error",
        "is_error": "is_error",
        "gasprice": "gas_price",
        "gas_price": "gas_price",
        "gasused": "gas_used",
        "gas_used": "gas_used",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    return df


def _parse_label(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(np.int64)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() >= 0.8:
        return numeric.fillna(0).astype(int)

    text = series.fillna("").astype(str).str.lower().str.strip()

    positive = {
        "1",
        "true",
        "yes",
        "hack",
        "hacker",
        "malicious",
        "fraud",
        "positive",
    }

    return text.isin(positive).astype(np.int64)


def _parse_timestamp_seconds(series: pd.Series) -> pd.Series:
    timestamp = pd.to_numeric(series, errors="coerce")

    # Support Unix milliseconds if needed.
    if timestamp.notna().any() and timestamp.dropna().median() > 10_000_000_000:
        timestamp = timestamp / 1000.0

    return timestamp.astype(np.float64)


def _build_unified_hack_df(
    csv_path: str,
    dataset_name: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, int, int, int]:
    """
    Leakage-safe choices:
        - hash is dropped because it is a unique transaction ID.
        - timeStamp is kept only as timestamp, not an edge feature.
        - blockNumber is used only as a sorting tie-breaker, not an edge feature.
        - from/to are remapped to src/dst node IDs.
        - numerical edge features are scaled using training rows only.
        - categorical edge features are encoded using training rows only.
    """
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    df = _prepare_hack_columns(df)

    required = ["src", "dst", "amount", "timestamp", "label"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name} missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    df = df.dropna(subset=required).copy()

    df["src"] = df["src"].astype(str).str.lower().str.strip()
    df["dst"] = df["dst"].astype(str).str.lower().str.strip()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["timestamp"] = _parse_timestamp_seconds(df["timestamp"])
    df["label"] = _parse_label(df["label"])

    if "block_number" in df.columns:
        df["block_number"] = pd.to_numeric(df["block_number"], errors="coerce")
    else:
        df["block_number"] = 0.0

    df["date_time"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        origin="unix",
        errors="coerce",
        utc=True,
    )

    df = df.dropna(
        subset=["src", "dst", "amount", "timestamp", "date_time", "label"]
    ).copy()

    df = df[(df["src"] != "") & (df["dst"] != "")].copy()

    df = df.sort_values(["date_time", "block_number"]).reset_index(drop=True)

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

    # Numerical edge features.
    df["amount"] = np.log1p(df["amount"].clip(lower=0))

    numeric_feature_cols = ["amount"]

    if "gas_price" in df.columns:
        df["gas_price"] = pd.to_numeric(df["gas_price"], errors="coerce")
        df["gas_price"] = np.log1p(df["gas_price"].clip(lower=0))
        numeric_feature_cols.append("gas_price")

    if "gas_used" in df.columns:
        df["gas_used"] = pd.to_numeric(df["gas_used"], errors="coerce")
        df["gas_used"] = np.log1p(df["gas_used"].clip(lower=0))
        numeric_feature_cols.append("gas_used")

    for column in numeric_feature_cols:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .astype(np.float32)
        )

    scaler = StandardScaler()

    if train_end > 0:
        df.loc[: train_end - 1, numeric_feature_cols] = scaler.fit_transform(
            df.loc[: train_end - 1, numeric_feature_cols]
        )

    if train_end < len(df):
        df.loc[train_end:, numeric_feature_cols] = scaler.transform(
            df.loc[train_end:, numeric_feature_cols]
        )

    # Categorical edge features.
    cat_cols = [
        "token_symbol",
        "contract_address",
    ]
    cat_cols = [column for column in cat_cols if column in df.columns]

    if cat_cols:
        df[cat_cols] = df[cat_cols].astype(str).fillna("__nan__")

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        if train_end > 0:
            encoder.fit(df.loc[: train_end - 1, cat_cols])
            df.loc[:, cat_cols] = encoder.transform(df[cat_cols])
        else:
            df.loc[:, cat_cols] = 0.0

        for column in cat_cols:
            df[column] = (
                pd.to_numeric(df[column], errors="coerce")
                .fillna(-1)
                .astype(np.float32)
            )

    ordered_cols = (
        ["src", "dst", "timestamp", "amount", "label"]
        + [column for column in ["gas_price", "gas_used"] if column in df.columns]
        + cat_cols
    )

    df = df[ordered_cols].copy()

    num_nodes = int(max(df["src"].max(), df["dst"].max()) + 1)

    edge_features = [
        column
        for column in ordered_cols
        if column not in {"src", "dst", "timestamp", "label","is_error"}
    ]

    print(
        f"[{dataset_name}] nodes={num_nodes}, edges={len(df)}, "
        f"positive_rate={df['label'].mean():.6f}, edge_features={edge_features}"
    )

    return df, num_nodes, train_end, val_end


def build_unified_ascendexhacker_df(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, int, int, int]:
    return _build_unified_hack_df(
        csv_path=csv_path,
        dataset_name="AscendEXHacker",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_rows=max_rows,
    )


def build_unified_upbithack_df(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_rows: Optional[int] = None,
) -> tuple[pd.DataFrame, int, int, int]:
    return _build_unified_hack_df(
        csv_path=csv_path,
        dataset_name="UpbitHack",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_rows=max_rows,
    )
