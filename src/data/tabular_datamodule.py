from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.splits import temporal_split_indices


AMLSIM_NAMES = {
    "amlsim",
    "aml-sim",
}

SAMLD_NAMES = {
    "samld",
    "saml-d",
}

AMLWORLD_NAMES = {
    "amlworld",
    "aml-world",
    "amlworld_hi_small",
    "amlworld-hi-small",
    "hi-small",
}

HACK_NAMES = {
    "ascendexhacker",
    "ascendex-hacker",
    "ascendex",
    "upbithack",
    "upbit-hack",
    "upbit",
}

AMLWORLD_PAIR_COLUMNS = [
    "__from_bank_key__",
    "__from_account_key__",
    "__to_bank_key__",
    "__to_account_key__",
]

HACK_PAIR_COLUMNS = [
    "__from_key__",
    "__to_key__",
]


class TabularDataset(Dataset):
    def __init__(
        self,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        y: np.ndarray,
    ):
        self.x_num = torch.tensor(
            x_num,
            dtype=torch.float32,
        )
        self.x_cat = torch.tensor(
            x_cat,
            dtype=torch.long,
        )
        self.y = torch.tensor(
            y,
            dtype=torch.float32,
        ).reshape(-1)

        size = self.y.size(0)

        if self.x_num.ndim != 2 or self.x_num.size(0) != size:
            raise ValueError(
                f"Invalid x_num shape: {tuple(self.x_num.shape)}"
            )

        if self.x_cat.ndim != 2 or self.x_cat.size(0) != size:
            raise ValueError(
                f"Invalid x_cat shape: {tuple(self.x_cat.shape)}"
            )

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, index):
        return {
            "x_num": self.x_num[index],
            "x_cat": self.x_cat[index],
            "y": self.y[index],
        }


@dataclass
class PreparedData:
    frame: pd.DataFrame
    numeric_columns: list[str]
    categorical_columns: list[str]


def _normalize_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _normalize_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        _normalize_column_name(column)
        for column in frame.columns
    ]
    return frame


def _rename_first_available(
    frame: pd.DataFrame,
    target: str,
    candidates: list[str],
) -> pd.DataFrame:
    if target in frame.columns:
        return frame

    for candidate in candidates:
        if candidate in frame.columns:
            return frame.rename(
                columns={candidate: target}
            )

    return frame


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing columns {missing}. "
            f"Available columns: {list(frame.columns)}"
        )


def _to_numeric(
    series: pd.Series,
) -> pd.Series:
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):
        series = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.strip()
        )

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _parse_label(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if float(numeric.notna().mean()) >= 0.8:
        return (
            numeric
            .where(numeric.isin([0, 1]))
            .astype(np.float32)
        )

    normalized = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    positive_values = {
        "1",
        "true",
        "yes",
        "y",
        "fraud",
        "fraudulent",
        "laundering",
        "is_laundering",
        "hack",
        "hacker",
        "malicious",
        "positive",
    }

    negative_values = {
        "0",
        "false",
        "no",
        "n",
        "normal",
        "benign",
        "negative",
    }

    output = pd.Series(
        np.nan,
        index=series.index,
        dtype=np.float32,
    )

    output.loc[
        normalized.isin(positive_values)
    ] = 1.0

    output.loc[
        normalized.isin(negative_values)
    ] = 0.0

    return output


def _add_time_features(
    frame: pd.DataFrame,
    timestamp_candidates: list[str],
    date_column: str | None = None,
    time_column: str | None = None,
) -> pd.DataFrame:
    """
    Add a sorting key and calendar features.

    __timestamp_sort__ is used only for chronological splitting.
    The absolute timestamp is not passed to TabAML.
    """
    frame = frame.copy()
    raw_timestamp: pd.Series | None = None

    if (
        date_column is not None
        and time_column is not None
        and date_column in frame.columns
        and time_column in frame.columns
    ):
        raw_timestamp = (
            frame[date_column]
            .fillna("")
            .astype(str)
            .str.strip()
            + " "
            + frame[time_column]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        for candidate in timestamp_candidates:
            if candidate in frame.columns:
                raw_timestamp = frame[candidate]
                break

    if raw_timestamp is None:
        raise ValueError(
            "No usable timestamp column was found."
        )

    numeric_timestamp = _to_numeric(
        raw_timestamp
    )

    if float(
        numeric_timestamp.notna().mean()
    ) >= 0.95:
        valid = numeric_timestamp.notna()
        frame = frame.loc[valid].copy()

        frame["__timestamp_sort__"] = (
            numeric_timestamp.loc[valid]
            .astype(np.float64)
        )

        # A numeric simulation step is safe for ordering, but it is not
        # interpreted as an absolute calendar timestamp.
        frame["hour"] = np.int64(0)
        frame["day_of_week"] = np.int64(0)
        frame["day_of_month"] = np.int64(0)
        frame["month"] = np.int64(0)
        frame["is_weekend"] = np.int64(0)

        return frame

    parsed = pd.to_datetime(
        raw_timestamp,
        errors="coerce",
        utc=True,
    )

    valid = parsed.notna()
    frame = frame.loc[valid].copy()
    parsed = parsed.loc[valid]

    if len(frame) == 0:
        raise ValueError(
            "All timestamp values are invalid."
        )

    frame["__timestamp_sort__"] = (
        parsed.astype("int64")
        / 1_000_000_000
    ).astype(np.float64)

    frame["hour"] = (
        parsed.dt.hour
        .astype(np.int64)
    )

    frame["day_of_week"] = (
        parsed.dt.dayofweek
        .astype(np.int64)
    )

    frame["day_of_month"] = (
        parsed.dt.day
        .astype(np.int64)
    )

    frame["month"] = (
        parsed.dt.month
        .astype(np.int64)
    )

    frame["is_weekend"] = (
        parsed.dt.dayofweek >= 5
    ).astype(np.int64)

    return frame


def _add_unix_time_features(
    frame: pd.DataFrame,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Add calendar features for datasets where the timestamp is a Unix
    timestamp, such as AscendEXHacker and UpbitHack.

    __timestamp_sort__ is still used only for chronological splitting.
    The absolute timestamp is not passed to TabAML.
    """
    frame = frame.copy()

    timestamp = _to_numeric(
        frame[timestamp_column]
    )

    valid = timestamp.notna()
    frame = frame.loc[valid].copy()
    timestamp = timestamp.loc[valid].astype(np.float64)

    if len(frame) == 0:
        raise ValueError(
            "All timestamp values are invalid."
        )

    median_timestamp = float(
        timestamp.median()
    )

    if median_timestamp > 10_000_000_000:
        timestamp_seconds = timestamp / 1000.0
    else:
        timestamp_seconds = timestamp

    parsed = pd.to_datetime(
        timestamp_seconds,
        unit="s",
        origin="unix",
        errors="coerce",
        utc=True,
    )

    valid_time = parsed.notna()
    frame = frame.loc[valid_time].copy()
    timestamp_seconds = timestamp_seconds.loc[valid_time]
    parsed = parsed.loc[valid_time]

    if len(frame) == 0:
        raise ValueError(
            "All Unix timestamp values are invalid."
        )

    frame["__timestamp_sort__"] = (
        timestamp_seconds
        .astype(np.float64)
    )

    frame["hour"] = (
        parsed.dt.hour
        .astype(np.int64)
    )

    frame["day_of_week"] = (
        parsed.dt.dayofweek
        .astype(np.int64)
    )

    frame["day_of_month"] = (
        parsed.dt.day
        .astype(np.int64)
    )

    frame["month"] = (
        parsed.dt.month
        .astype(np.int64)
    )

    frame["is_weekend"] = (
        parsed.dt.dayofweek >= 5
    ).astype(np.int64)

    return frame


def _log_amount(
    series: pd.Series,
) -> pd.Series:
    values = (
        _to_numeric(series)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .clip(lower=0)
    )

    return np.log1p(
        values
    ).astype(np.float32)


def _clean_pair_component(
    series: pd.Series,
) -> pd.Series:
    return (
        series.fillna("__missing__")
        .astype(str)
        .str.strip()
        .str.lower()
        .replace("", "__missing__")
    )


def _pair_hash(
    frame: pd.DataFrame,
    pair_columns: list[str],
    dataset_name: str,
) -> np.ndarray:
    _require_columns(
        frame,
        pair_columns,
        f"{dataset_name} pair filtering",
    )

    pair_frame = frame[
        pair_columns
    ].copy()

    for column in pair_columns:
        pair_frame[column] = (
            _clean_pair_component(
                pair_frame[column]
            )
        )

    return pd.util.hash_pandas_object(
        pair_frame,
        index=False,
    ).to_numpy(
        dtype=np.uint64
    )


def _make_pair_disjoint(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    pair_columns: list[str],
    dataset_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Enforce strict endpoint-pair separation.

    - Remove from train every ordered tuple seen in validation or test.
    - Remove from validation every ordered tuple seen in test.
    - Keep test unchanged.

    Labels from validation and test are never used.
    """
    train_hashes = _pair_hash(
        train_frame,
        pair_columns,
        dataset_name,
    )

    val_hashes = _pair_hash(
        val_frame,
        pair_columns,
        dataset_name,
    )

    test_hashes = _pair_hash(
        test_frame,
        pair_columns,
        dataset_name,
    )

    future_hashes = np.union1d(
        np.unique(val_hashes),
        np.unique(test_hashes),
    )

    remove_train = np.isin(
        train_hashes,
        future_hashes,
    )

    remove_val = np.isin(
        val_hashes,
        np.unique(test_hashes),
    )

    train_frame = (
        train_frame.loc[~remove_train]
        .copy()
        .reset_index(drop=True)
    )

    val_frame = (
        val_frame.loc[~remove_val]
        .copy()
        .reset_index(drop=True)
    )

    test_frame = (
        test_frame
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"[TabAML][{dataset_name}PairDisjoint] "
        f"removed_train={int(remove_train.sum())}, "
        f"removed_val={int(remove_val.sum())}"
    )

    train_unique = np.unique(
        _pair_hash(
            train_frame,
            pair_columns,
            dataset_name,
        )
    )

    val_unique = np.unique(
        _pair_hash(
            val_frame,
            pair_columns,
            dataset_name,
        )
    )

    test_unique = np.unique(
        _pair_hash(
            test_frame,
            pair_columns,
            dataset_name,
        )
    )

    if (
        np.intersect1d(
            train_unique,
            val_unique,
        ).size > 0
        or np.intersect1d(
            train_unique,
            test_unique,
        ).size > 0
        or np.intersect1d(
            val_unique,
            test_unique,
        ).size > 0
    ):
        raise RuntimeError(
            f"{dataset_name} pair-disjoint verification failed."
        )

    print(
        f"[TabAML][{dataset_name}PairDisjoint] "
        "verification passed."
    )

    return (
        train_frame,
        val_frame,
        test_frame,
    )


def _prepare_amlsim(
    raw_frame: pd.DataFrame,
) -> PreparedData:
    frame = _normalize_columns(
        raw_frame
    )

    frame = _rename_first_available(
        frame,
        "amount",
        [
            "tx_amount",
            "transaction_amount",
            "amount_paid",
            "amount_received",
        ],
    )

    frame = _rename_first_available(
        frame,
        "label",
        [
            "is_fraud",
            "is_laundering",
            "fraud",
            "target",
        ],
    )

    timestamp_column = next(
        (
            column
            for column in [
                "timestamp",
                "time",
                "transaction_time",
                "step",
            ]
            if column in frame.columns
        ),
        None,
    )

    if timestamp_column is None:
        raise ValueError(
            "AMLSim requires a timestamp, time, transaction_time, or step column."
        )

    _require_columns(
        frame,
        [
            "amount",
            "label",
            timestamp_column,
        ],
        "AMLSim",
    )

    frame = _add_time_features(
        frame,
        [timestamp_column],
    )

    frame["amount_log"] = (
        _log_amount(
            frame["amount"]
        )
    )

    frame["label"] = (
        _parse_label(
            frame["label"]
        )
    )

    frame = frame.dropna(
        subset=[
            "label",
            "amount_log",
        ]
    ).copy()

    numeric_columns = [
        "amount_log",
    ]

    categorical_columns = [
        column
        for column in [
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
            "tx_type",
            "transaction_type",
            "currency",
            "payment_currency",
            "payment_type",
        ]
        if column in frame.columns
    ]

    keep_columns = (
        [
            "label",
            "__timestamp_sort__",
        ]
        + numeric_columns
        + categorical_columns
    )

    return PreparedData(
        frame=frame[keep_columns].copy(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def _prepare_samld(
    raw_frame: pd.DataFrame,
) -> PreparedData:
    frame = _normalize_columns(
        raw_frame
    )

    frame = _rename_first_available(
        frame,
        "amount",
        [
            "transaction_amount",
            "amount",
            "amt",
            "tx_amount",
            "amount_paid",
            "amount_received",
        ],
    )

    frame = _rename_first_available(
        frame,
        "label",
        [
            "is_laundering",
            "is_fraud",
            "fraud",
            "target",
        ],
    )

    _require_columns(
        frame,
        [
            "amount",
            "label",
        ],
        "SAML-D",
    )

    date_column = next(
        (
            column
            for column in [
                "date",
                "transaction_date",
            ]
            if column in frame.columns
        ),
        None,
    )

    time_column = next(
        (
            column
            for column in [
                "time",
                "transaction_time",
            ]
            if column in frame.columns
        ),
        None,
    )

    timestamp_candidates = [
        column
        for column in [
            "timestamp",
            "date_time",
            "datetime",
            "transaction_datetime",
        ]
        if column in frame.columns
    ]

    if (
        date_column is None
        and not timestamp_candidates
    ):
        raise ValueError(
            "SAML-D requires a date/time or timestamp column."
        )

    frame = _add_time_features(
        frame,
        timestamp_candidates,
        date_column=date_column,
        time_column=time_column,
    )

    frame["amount_log"] = (
        _log_amount(
            frame["amount"]
        )
    )

    frame["label"] = (
        _parse_label(
            frame["label"]
        )
    )

    frame = frame.dropna(
        subset=[
            "label",
            "amount_log",
        ]
    ).copy()

    numeric_columns = [
        "amount_log",
    ]

    categorical_columns = [
        column
        for column in [
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
            "sender_bank_location",
            "receiver_bank_location",
            "payment_currency",
            "received_currency",
            "receiving_currency",
            "payment_type",
            "payment_format",
        ]
        if column in frame.columns
    ]

    keep_columns = (
        [
            "label",
            "__timestamp_sort__",
        ]
        + numeric_columns
        + categorical_columns
    )

    return PreparedData(
        frame=frame[keep_columns].copy(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def _prepare_amlworld(
    raw_frame: pd.DataFrame,
) -> PreparedData:
    """
    AMLWorld split-overlap detection uses:

        (
            From Bank,
            Account,
            To Bank,
            Account.1,
        )

    Account and Account.1 are never passed to TabAML.
    """
    frame = _normalize_columns(
        raw_frame
    )

    frame = _rename_first_available(
        frame,
        "label",
        [
            "is_laundering",
            "is_fraud",
            "fraud",
            "target",
        ],
    )

    _require_columns(
        frame,
        [
            "timestamp",
            "from_bank",
            "account",
            "to_bank",
            "account_1",
            "label",
        ],
        "AMLWorld",
    )

    if (
        "amount_received" not in frame.columns
        and "amount_paid" not in frame.columns
    ):
        raise ValueError(
            "AMLWorld requires amount_received or amount_paid."
        )

    frame[
        "__from_bank_key__"
    ] = _clean_pair_component(
        frame["from_bank"]
    )

    frame[
        "__from_account_key__"
    ] = _clean_pair_component(
        frame["account"]
    )

    frame[
        "__to_bank_key__"
    ] = _clean_pair_component(
        frame["to_bank"]
    )

    frame[
        "__to_account_key__"
    ] = _clean_pair_component(
        frame["account_1"]
    )

    frame = _add_time_features(
        frame,
        [
            "timestamp",
            "date_time",
            "datetime",
            "time",
            "transaction_time",
        ],
    )

    numeric_columns: list[str] = []

    if "amount_received" in frame.columns:
        frame[
            "amount_received_log"
        ] = _log_amount(
            frame["amount_received"]
        )

        numeric_columns.append(
            "amount_received_log"
        )

    if "amount_paid" in frame.columns:
        frame[
            "amount_paid_log"
        ] = _log_amount(
            frame["amount_paid"]
        )

        numeric_columns.append(
            "amount_paid_log"
        )

    if (
        "amount_received" in frame.columns
        and "amount_paid" in frame.columns
    ):
        frame[
            "amount_difference"
        ] = (
            _to_numeric(
                frame["amount_received"]
            )
            - _to_numeric(
                frame["amount_paid"]
            )
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).astype(np.float32)

        numeric_columns.append(
            "amount_difference"
        )

    frame["label"] = (
        _parse_label(
            frame["label"]
        )
    )

    frame = frame.dropna(
        subset=["label"]
    ).copy()

    categorical_columns = [
        column
        for column in [
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
            "from_bank",
            "to_bank",
            "receiving_currency",
            "received_currency",
            "payment_currency",
            "payment_format",
            "payment_type",
        ]
        if column in frame.columns
    ]

    keep_columns = (
        [
            "label",
            "__timestamp_sort__",
        ]
        + AMLWORLD_PAIR_COLUMNS
        + numeric_columns
        + categorical_columns
    )

    return PreparedData(
        frame=frame[keep_columns].copy(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def _prepare_hack_dataset(
    raw_frame: pd.DataFrame,
    dataset_name: str,
) -> PreparedData:
    """
    AscendEXHacker / UpbitHack.

    Raw columns:
        hash, from, to, value, timeStamp, blockNumber,
        tokenSymbol, contractAddress, isError, gasPrice, gasUsed, label

    TabAML features:
        numeric:
            value_log, gas_price_log, gas_used_log
        categorical:
            hour, day_of_week, day_of_month, month, is_weekend,
            token_symbol, contract_address, is_error

    Not model features:
        hash          transaction ID
        from, to      used only for pair-disjoint filtering
        timeStamp     used only for chronological splitting/calendar features
        blockNumber   used only as chronological tie-breaker
    """
    frame = _normalize_columns(
        raw_frame
    )

    rename_map = {
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
        if old in frame.columns and new not in frame.columns:
            frame = frame.rename(
                columns={old: new}
            )

    _require_columns(
        frame,
        [
            "from",
            "to",
            "value",
            "timestamp",
            "label",
        ],
        dataset_name,
    )

    frame["__from_key__"] = (
        _clean_pair_component(
            frame["from"]
        )
    )

    frame["__to_key__"] = (
        _clean_pair_component(
            frame["to"]
        )
    )

    frame = _add_unix_time_features(
        frame,
        "timestamp",
    )

    if "block_number" in frame.columns:
        block_number = _to_numeric(
            frame["block_number"]
        ).fillna(0.0)

        frame["__timestamp_sort__"] = (
            frame["__timestamp_sort__"].astype(np.float64)
            * 1_000_000_000
            + block_number.astype(np.float64)
        )

    frame["value_log"] = (
        _log_amount(
            frame["value"]
        )
    )

    numeric_columns = [
        "value_log",
    ]

    if "gas_price" in frame.columns:
        frame["gas_price_log"] = (
            _log_amount(
                frame["gas_price"]
            )
        )

        numeric_columns.append(
            "gas_price_log"
        )

    if "gas_used" in frame.columns:
        frame["gas_used_log"] = (
            _log_amount(
                frame["gas_used"]
            )
        )

        numeric_columns.append(
            "gas_used_log"
        )

    frame["label"] = (
        _parse_label(
            frame["label"]
        )
    )

    frame = frame.dropna(
        subset=[
            "label",
            "value_log",
        ]
    ).copy()

    categorical_columns = [
        column
        for column in [
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "is_weekend",
            "token_symbol",
            "contract_address",
            "is_error",
        ]
        if column in frame.columns
    ]

    keep_columns = (
        [
            "label",
            "__timestamp_sort__",
        ]
        + HACK_PAIR_COLUMNS
        + numeric_columns
        + categorical_columns
    )

    return PreparedData(
        frame=frame[keep_columns].copy(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


class TabularAMLDataModule(
    L.LightningDataModule
):
    """
    Leakage-controlled tabular datamodule.

    Controls:
        1. Stable chronological split.
        2. Pair-disjoint filtering before preprocessing.
        3. Numeric medians fitted on training only.
        4. StandardScaler fitted on training only.
        5. Category dictionaries fitted on training only.
        6. Rare/unseen validation/test categories map to zero.
        7. Entity identifiers are never model features.
        8. Absolute timestamps are never model features.
        9. Class weight is calculated from training labels only.
    """

    def __init__(
        self,
        dataset_name: str,
        csv_path: str,
        batch_size: int = 256,
        train_ratio: float = 0.75,
        val_ratio: float = 0.15,
        max_rows: Optional[int] = None,
        num_workers: int = 4,
        max_categories_per_feature: Optional[int] = 8192,
        min_category_frequency: int = 2,
        pair_disjoint: bool = True,
    ):
        super().__init__()

        if not 0 < train_ratio < 1:
            raise ValueError(
                "train_ratio must be in (0, 1)."
            )

        if not 0 <= val_ratio < 1:
            raise ValueError(
                "val_ratio must be in [0, 1)."
            )

        if train_ratio + val_ratio >= 1:
            raise ValueError(
                "train_ratio + val_ratio must be less than 1."
            )

        if min_category_frequency < 1:
            raise ValueError(
                "min_category_frequency must be at least 1."
            )

        if (
            max_categories_per_feature is not None
            and max_categories_per_feature < 2
        ):
            raise ValueError(
                "max_categories_per_feature must be at least 2."
            )

        self.dataset_name = (
            dataset_name.lower().strip()
        )

        self.csv_path = csv_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.max_rows = max_rows
        self.num_workers = num_workers

        self.max_categories_per_feature = (
            max_categories_per_feature
        )

        self.min_category_frequency = (
            min_category_frequency
        )

        self.pair_disjoint = pair_disjoint

        self.train_ds: TabularDataset | None = None
        self.val_ds: TabularDataset | None = None
        self.test_ds: TabularDataset | None = None

        self.num_numeric_features = 0
        self.cat_cardinalities: list[int] = []
        self.cat_feature_names: list[str] = []
        self.num_feature_names: list[str] = []

        self.numeric_fill_values: dict[
            str,
            float,
        ] = {}

        self.category_mappings: dict[
            str,
            dict[str, int],
        ] = {}

        self.scaler: StandardScaler | None = None
        self.train_pos_weight = 1.0

    def _load_raw(
        self,
    ) -> pd.DataFrame:
        path = Path(
            self.csv_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {path}"
            )

        return pd.read_csv(
            path,
            nrows=self.max_rows,
            low_memory=False,
        )

    def _prepare(
        self,
        raw_frame: pd.DataFrame,
    ) -> PreparedData:
        if self.dataset_name in AMLSIM_NAMES:
            return _prepare_amlsim(
                raw_frame
            )

        if self.dataset_name in SAMLD_NAMES:
            return _prepare_samld(
                raw_frame
            )

        if self.dataset_name in AMLWORLD_NAMES:
            return _prepare_amlworld(
                raw_frame
            )

        if self.dataset_name in HACK_NAMES:
            return _prepare_hack_dataset(
                raw_frame,
                self.dataset_name,
            )

        raise ValueError(
            f"Unsupported TabAML dataset: {self.dataset_name}"
        )

    @staticmethod
    def _clean_category(
        series: pd.Series,
    ) -> pd.Series:
        return (
            series.fillna("__missing__")
            .astype(str)
            .str.strip()
            .replace("", "__missing__")
        )

    def _fit_numeric_preprocessing(
        self,
        train_frame: pd.DataFrame,
        numeric_columns: list[str],
    ) -> None:
        self.numeric_fill_values = {}

        if not numeric_columns:
            self.scaler = None
            return

        train_numeric = train_frame[
            numeric_columns
        ].copy()

        for column in numeric_columns:
            values = (
                _to_numeric(
                    train_numeric[column]
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

            median = values.median()

            if pd.isna(median):
                median = 0.0

            median = float(median)

            self.numeric_fill_values[
                column
            ] = median

            train_numeric[column] = (
                values.fillna(median)
            )

        self.scaler = StandardScaler()

        self.scaler.fit(
            train_numeric.to_numpy(
                dtype=np.float64
            )
        )

    def _transform_numeric(
        self,
        frame: pd.DataFrame,
        numeric_columns: list[str],
    ) -> np.ndarray:
        if not numeric_columns:
            return np.empty(
                (len(frame), 0),
                dtype=np.float32,
            )

        if self.scaler is None:
            raise RuntimeError(
                "Numeric preprocessing has not been fitted."
            )

        numeric = frame[
            numeric_columns
        ].copy()

        for column in numeric_columns:
            values = (
                _to_numeric(
                    numeric[column]
                )
                .replace(
                    [np.inf, -np.inf],
                    np.nan,
                )
            )

            numeric[column] = values.fillna(
                self.numeric_fill_values[
                    column
                ]
            )

        return self.scaler.transform(
            numeric.to_numpy(
                dtype=np.float64
            )
        ).astype(
            np.float32
        )

    def _fit_category_mappings(
        self,
        train_frame: pd.DataFrame,
        categorical_columns: list[str],
    ) -> None:
        self.category_mappings = {}
        self.cat_cardinalities = []

        for column in categorical_columns:
            values = self._clean_category(
                train_frame[column]
            )

            values = values[
                values != "__missing__"
            ]

            counts = values.value_counts()

            retained = [
                (
                    str(category),
                    int(frequency),
                )
                for category, frequency in counts.items()
                if int(frequency)
                >= self.min_category_frequency
            ]

            retained.sort(
                key=lambda item: (
                    -item[1],
                    item[0],
                )
            )

            if (
                self.max_categories_per_feature
                is not None
            ):
                retained = retained[
                    : (
                        self.max_categories_per_feature
                        - 1
                    )
                ]

            mapping = {
                category: index + 1
                for index, (
                    category,
                    _,
                ) in enumerate(
                    retained
                )
            }

            self.category_mappings[
                column
            ] = mapping

            self.cat_cardinalities.append(
                len(mapping) + 1
            )

    def _transform_categories(
        self,
        frame: pd.DataFrame,
        categorical_columns: list[str],
    ) -> np.ndarray:
        if not categorical_columns:
            return np.empty(
                (len(frame), 0),
                dtype=np.int64,
            )

        encoded_columns = []

        for column in categorical_columns:
            encoded = (
                self._clean_category(
                    frame[column]
                )
                .map(
                    self.category_mappings[
                        column
                    ]
                )
                .fillna(0)
                .astype(np.int64)
                .to_numpy()
            )

            encoded_columns.append(
                encoded
            )

        return np.stack(
            encoded_columns,
            axis=1,
        )

    def _make_dataset(
        self,
        frame: pd.DataFrame,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> TabularDataset:
        x_num = self._transform_numeric(
            frame,
            numeric_columns,
        )

        x_cat = self._transform_categories(
            frame,
            categorical_columns,
        )

        y = frame[
            "label"
        ].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        return TabularDataset(
            x_num=x_num,
            x_cat=x_cat,
            y=y,
        )

    def setup(
        self,
        stage=None,
    ):
        if self.train_ds is not None:
            return

        prepared = self._prepare(
            self._load_raw()
        )

        frame = (
            prepared.frame
            .sort_values(
                "__timestamp_sort__",
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        total_rows = len(frame)

        train_end, val_end = (
            temporal_split_indices(
                total_rows,
                self.train_ratio,
                self.val_ratio,
            )
        )

        train_frame = frame.iloc[
            :train_end
        ].copy()

        val_frame = frame.iloc[
            train_end:val_end
        ].copy()

        test_frame = frame.iloc[
            val_end:
        ].copy()

        print(
            "[TabAML] chronological split before filtering: "
            f"train={len(train_frame)}, "
            f"val={len(val_frame)}, "
            f"test={len(test_frame)}"
        )

        if (
            self.pair_disjoint
            and self.dataset_name
            in AMLWORLD_NAMES
        ):
            (
                train_frame,
                val_frame,
                test_frame,
            ) = _make_pair_disjoint(
                train_frame,
                val_frame,
                test_frame,
                AMLWORLD_PAIR_COLUMNS,
                "AMLWorld",
            )

        if (
            self.pair_disjoint
            and self.dataset_name
            in HACK_NAMES
        ):
            (
                train_frame,
                val_frame,
                test_frame,
            ) = _make_pair_disjoint(
                train_frame,
                val_frame,
                test_frame,
                HACK_PAIR_COLUMNS,
                self.dataset_name,
            )

        if min(
            len(train_frame),
            len(val_frame),
            len(test_frame),
        ) == 0:
            raise ValueError(
                "A split became empty after filtering."
            )

        numeric_columns = list(
            prepared.numeric_columns
        )

        # Drop categorical features that are constant in the filtered
        # training split. This decision uses training data only.
        categorical_columns = [
            column
            for column
            in prepared.categorical_columns
            if self._clean_category(
                train_frame[column]
            ).nunique() > 1
        ]

        self._fit_numeric_preprocessing(
            train_frame,
            numeric_columns,
        )

        self._fit_category_mappings(
            train_frame,
            categorical_columns,
        )

        self.num_numeric_features = len(
            numeric_columns
        )

        self.num_feature_names = list(
            numeric_columns
        )

        self.cat_feature_names = list(
            categorical_columns
        )

        labels = train_frame[
            "label"
        ].to_numpy(
            dtype=np.float32
        )

        positives = int(
            (labels == 1).sum()
        )

        negatives = int(
            (labels == 0).sum()
        )

        if positives == 0:
            raise ValueError(
                "The filtered training split contains no positive examples."
            )

        self.train_pos_weight = (
            negatives / positives
        )

        self.train_ds = self._make_dataset(
            train_frame,
            numeric_columns,
            categorical_columns,
        )

        self.val_ds = self._make_dataset(
            val_frame,
            numeric_columns,
            categorical_columns,
        )

        self.test_ds = self._make_dataset(
            test_frame,
            numeric_columns,
            categorical_columns,
        )

        print(
            "[TabAML] final split: "
            f"train={len(train_frame)}, "
            f"val={len(val_frame)}, "
            f"test={len(test_frame)}"
        )

        print(
            "[TabAML] positive rates: "
            f"train={train_frame['label'].mean():.6f}, "
            f"val={val_frame['label'].mean():.6f}, "
            f"test={test_frame['label'].mean():.6f}"
        )

        print(
            f"[TabAML] numeric columns: "
            f"{self.num_feature_names}"
        )

        print(
            f"[TabAML] categorical columns: "
            f"{self.cat_feature_names}"
        )

        print(
            f"[TabAML] categorical cardinalities: "
            f"{self.cat_cardinalities}"
        )

        print(
            f"[TabAML] training-only pos_weight: "
            f"{self.train_pos_weight:.6f}"
        )

        if self.dataset_name in AMLWORLD_NAMES:
            print(
                "[TabAML][AMLWorld] Account and Account.1 "
                "are used only for pair-overlap filtering, "
                "not as model features."
            )

        if self.dataset_name in HACK_NAMES:
            print(
                "[TabAML][HackDataset] hash, from, to, timeStamp, "
                "and blockNumber are not model features. "
                "from/to are used only for pair-overlap filtering."
            )

    def _loader(
        self,
        dataset,
        shuffle: bool,
    ):
        if dataset is None:
            raise RuntimeError(
                "Call setup() before requesting a DataLoader."
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=(
                self.num_workers > 0
            ),
            pin_memory=(
                torch.cuda.is_available()
            ),
            drop_last=False,
        )

    def train_dataloader(self):
        return self._loader(
            self.train_ds,
            True,
        )

    def val_dataloader(self):
        return self._loader(
            self.val_ds,
            False,
        )

    def test_dataloader(self):
        return self._loader(
            self.test_ds,
            False,
        )