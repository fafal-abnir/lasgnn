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

    def _add_calendar_features(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        if np.issubdtype(df[timestamp_col].dtype, np.number):
            ts_num = pd.to_numeric(df[timestamp_col], errors="coerce")
            dt = pd.to_datetime(ts_num, unit="s", origin="unix", errors="coerce", utc=True)
        else:
            dt = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)

        df = df.copy()
        df["__ts_num__"] = pd.to_numeric(
            pd.Series(dt.astype("int64") / 1e9, index=df.index),
            errors="coerce",
        )

        df = df.dropna(subset=["__ts_num__"]).copy()
        dt = pd.to_datetime(df["__ts_num__"], unit="s", utc=True)

        df["day"] = dt.dt.day.astype(np.int64)
        df["month"] = dt.dt.month.astype(np.int64)
        df["year"] = dt.dt.year.astype(np.int64)
        df["hour"] = dt.dt.hour.astype(np.int64)
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(np.int64)
        df["day_of_week"] = dt.dt.dayofweek.astype(np.int64)

        return df

    def _prepare_samld(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
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

        required = ["amount", "label", time_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"SAML-D missing required columns: {missing}. Found: {list(df.columns)}")

        if self.max_rows is not None:
            df = df.iloc[: self.max_rows].copy()

        df = df.dropna(subset=required).copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        df = self._add_calendar_features(df, time_col)
        df = df.dropna(subset=["amount"]).copy()

        df = df.sort_values("__ts_num__").reset_index(drop=True)
        df["amount"] = np.log1p(df["amount"].clip(lower=0))

        keep_cols = [
            "amount",
            "label",
            "day",
            "month",
            "year",
            "hour",
            "is_weekend",
            "day_of_week",
        ]
        safe_optional_cat = [
            "sender_bank_location",
            "receiver_bank_location",
            "payment_currency",
            "received_currency",
            "payment_type",
        ]
        keep_cols += [c for c in safe_optional_cat if c in df.columns]
        keep_cols += ["__ts_num__"]

        df = df[keep_cols].copy()
        return df

    def _prepare_amlsim(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_map = {
            "tx_amount": "amount",
            "time": "timestamp",
            "is_fraud": "label",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})

        required = ["amount", "label", "timestamp"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"AMLSim missing required columns: {missing}. Found: {list(df.columns)}")

        if self.max_rows is not None:
            df = df.iloc[: self.max_rows].copy()

        df = df.dropna(subset=required).copy()
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        df = self._add_calendar_features(df, "timestamp")
        df = df.dropna(subset=["amount"]).copy()

        df = df.sort_values("__ts_num__").reset_index(drop=True)
        df["amount"] = np.log1p(df["amount"].clip(lower=0))

        # STRICT AMLSim whitelist to avoid leakage
        keep_cols = [
            "amount",
            "label",
            "day",
            "month",
            "year",
            "hour",
            "is_weekend",
            "day_of_week",
        ]
        safe_optional_cat = [
            "tx_type",
            "transaction_type",
            "currency",
        ]
        keep_cols += [c for c in safe_optional_cat if c in df.columns]
        keep_cols += ["__ts_num__"]

        df = df[keep_cols].copy()
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

        categorical_candidates = [
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
        numeric_cols = ["amount"] if "amount" in df.columns else []

        x_cat_arrays = []
        cat_cardinalities = []
        cat_feature_names = []

        for col in categorical_cols:
            vals = df[col].astype(str).fillna("__nan__")
            le = LabelEncoder()
            enc = le.fit_transform(vals).astype(np.int64)
            x_cat_arrays.append(enc)
            cat_cardinalities.append(int(enc.max()) + 1)
            cat_feature_names.append(col)

        if len(x_cat_arrays) == 0:
            x_cat = np.zeros((len(df), 0), dtype=np.int64)
        else:
            x_cat = np.stack(x_cat_arrays, axis=1).astype(np.int64)

        x_num = (
            df[numeric_cols].to_numpy(dtype=np.float32)
            if len(numeric_cols) > 0
            else np.zeros((len(df), 0), dtype=np.float32)
        )
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

        print(f"[TabAML] train={train_end}, val={val_end - train_end}, test={n - val_end}")
        print(f"[TabAML] positive rates: "
              f"train={y[:train_end].mean():.6f}, "
              f"val={y[train_end:val_end].mean():.6f}, "
              f"test={y[val_end:].mean():.6f}")
        print(f"[TabAML] categorical columns: {categorical_cols}")
        print(f"[TabAML] numeric columns: {numeric_cols}")

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