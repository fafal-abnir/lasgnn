from __future__ import annotations

from typing import Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.splits import temporal_split_indices


class TabularDataset(Dataset):
    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray):
        self.x_num = torch.tensor(x_num, dtype=torch.float)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x_num": self.x_num[idx],
            "x_cat": self.x_cat[idx],
            "y": self.y[idx],
        }


class TabularAMLDataModule(L.LightningDataModule):
    """
    A more faithful Tab-AML style tabular preprocessing path.

    The paper is on SAML-D, but this module also works on AMLSim if columns exist.
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
    ):
        super().__init__()
        self.dataset_name = dataset_name.lower()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.max_rows = max_rows
        self.num_workers = num_workers

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.num_numeric_features = 0
        self.cat_cardinalities: list[int] = []
        self.cat_feature_names: list[str] = []

    def _load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.lower().str.strip()
        return df

    def _prepare_samld(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "sender_account": "sender_account",
            "receiver_account": "receiver_account",
            "transaction_amount": "amount",
            "amt": "amount",
            "tx_amount": "amount",
            "is_laundering": "label",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        if "date" in df.columns:
            time_col = "date"
        elif "time" in df.columns:
            time_col = "time"
        elif "timestamp" in df.columns:
            time_col = "timestamp"
        else:
            raise ValueError("SAML-D requires one of: date, time, timestamp")

        required = ["sender_account", "receiver_account", "amount", "label", time_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"SAML-D missing required columns: {missing}. Found: {list(df.columns)}")

        if self.max_rows is not None:
            df = df.iloc[: self.max_rows].copy()

        df = df.dropna(subset=required).copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        if ts.isna().all():
            df["timestamp"] = pd.to_numeric(df[time_col], errors="coerce")
        else:
            df["timestamp"] = ts.astype("int64") / 1e9

        df = df.dropna(subset=["amount", "timestamp"]).copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Temporal feature engineering faithful to paper spirit.
        dt = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["day"] = dt.dt.day.astype(np.int64)
        df["month"] = dt.dt.month.astype(np.int64)
        df["year"] = dt.dt.year.astype(np.int64)
        df["hour"] = dt.dt.hour.astype(np.int64)
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.int64)
        df["day_of_week"] = dt.dt.dayofweek.astype(np.int64)

        # log transform amount
        df["amount"] = np.log1p(df["amount"].clip(lower=0))

        # Drop original temporal field(s) and typology if present
        drop_cols = [c for c in [time_col, "timestamp", "laundering_type", "typology_classification"] if c in df.columns]
        # keep timestamp-derived features only, not raw timestamp
        df = df.drop(columns=drop_cols, errors="ignore")

        return df

    def _prepare_amlsim(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "sender_account_id": "sender_account",
            "receiver_account_id": "receiver_account",
            "tx_amount": "amount",
            "time": "timestamp",
            "is_fraud": "label",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        required = ["sender_account", "receiver_account", "amount", "label", "timestamp"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"AMLSim missing required columns: {missing}. Found: {list(df.columns)}")

        if self.max_rows is not None:
            df = df.iloc[: self.max_rows].copy()

        df = df.dropna(subset=required).copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

        if np.issubdtype(df["timestamp"].dtype, np.number):
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            dt = pd.to_datetime(df["timestamp"], unit="s", errors="coerce", utc=True)
            if dt.isna().all():
                # fallback for AMLSim step-like timestamps
                # convert relative step into synthetic datetime anchored at epoch
                dt = pd.to_datetime(df["timestamp"], unit="s", origin="unix", utc=True)
        else:
            dt = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
            df["timestamp"] = dt.astype("int64") / 1e9

        df = df.dropna(subset=["amount", "timestamp"]).copy()
        df = df.sort_values("timestamp").reset_index(drop=True)

        df["day"] = dt.dt.day.fillna(0).astype(np.int64)
        df["month"] = dt.dt.month.fillna(0).astype(np.int64)
        df["year"] = dt.dt.year.fillna(0).astype(np.int64)
        df["hour"] = dt.dt.hour.fillna(0).astype(np.int64)
        df["is_weekend"] = (dt.dt.dayofweek.fillna(0) >= 5).astype(np.int64)
        df["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(np.int64)

        df["amount"] = np.log1p(df["amount"].clip(lower=0))

        drop_cols = [c for c in ["timestamp", "transaction_timestamp", "datetime"] if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")
        return df

    def setup(self, stage=None):
        df = self._load_raw()

        if self.dataset_name in {"samld", "saml-d"}:
            df = self._prepare_samld(df)
        elif self.dataset_name == "amlsim":
            df = self._prepare_amlsim(df)
        else:
            raise ValueError(f"Unsupported dataset for Tab-AML path: {self.dataset_name}")

        n = len(df)
        train_end, val_end = temporal_split_indices(n, self.train_ratio, self.val_ratio)

        # faithful categorical choices: account IDs, locations, currencies, payment types, calendar fields
        categorical_candidates = [
            "sender_account",
            "receiver_account",
            "sender_bank_location",
            "receiver_bank_location",
            "payment_currency",
            "received_currency",
            "payment_type",
            "tx_type",
            "transaction_type",
            "currency",
            "day",
            "month",
            "year",
            "hour",
            "is_weekend",
            "day_of_week",
        ]

        categorical_cols = [c for c in categorical_candidates if c in df.columns]
        numeric_cols = ["amount"]

        # any remaining non-label columns not used yet:
        protected = set(categorical_cols + numeric_cols + ["label"])
        remaining = [c for c in df.columns if c not in protected]

        # encode object-like remaining cols as categorical; numeric remaining as continuous
        for c in remaining:
            if df[c].dtype == object:
                categorical_cols.append(c)
            elif np.issubdtype(df[c].dtype, np.number):
                numeric_cols.append(c)

        cat_arrays = []
        cat_cardinalities = []
        cat_feature_names = []

        for col in categorical_cols:
            vals = df[col].astype(str).fillna("__nan__")
            le = LabelEncoder()
            enc = le.fit_transform(vals).astype(np.int64)
            cat_arrays.append(enc)
            cat_cardinalities.append(int(enc.max()) + 1)
            cat_feature_names.append(col)

        if len(cat_arrays) == 0:
            x_cat = np.zeros((len(df), 0), dtype=np.int64)
        else:
            x_cat = np.stack(cat_arrays, axis=1).astype(np.int64)

        x_num = df[numeric_cols].to_numpy(dtype=np.float32) if len(numeric_cols) > 0 else np.zeros((len(df), 0), dtype=np.float32)
        y = df["label"].to_numpy(dtype=np.float32)

        scaler = StandardScaler()
        if x_num.shape[1] > 0:
            x_num_train = scaler.fit_transform(x_num[:train_end]).astype(np.float32)
            x_num_val = scaler.transform(x_num[train_end:val_end]).astype(np.float32)
            x_num_test = scaler.transform(x_num[val_end:]).astype(np.float32)
        else:
            x_num_train = x_num[:train_end]
            x_num_val = x_num[train_end:val_end]
            x_num_test = x_num[val_end:]

        self.train_ds = TabularDataset(x_num_train, x_cat[:train_end], y[:train_end])
        self.val_ds = TabularDataset(x_num_val, x_cat[train_end:val_end], y[train_end:val_end])
        self.test_ds = TabularDataset(x_num_test, x_cat[val_end:], y[val_end:])

        self.num_numeric_features = x_num.shape[1]
        self.cat_cardinalities = cat_cardinalities
        self.cat_feature_names = cat_feature_names

    def _loader(self, ds, shuffle: bool):
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, True)

    def val_dataloader(self):
        return self._loader(self.val_ds, False)

    def test_dataloader(self):
        return self._loader(self.test_ds, False)