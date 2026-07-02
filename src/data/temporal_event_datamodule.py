from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.unified import load_unified_df


@dataclass
class TemporalEvents:
    src: torch.Tensor
    dst: torch.Tensor
    t: torch.Tensor
    y: torch.Tensor
    msg: torch.Tensor
    x: torch.Tensor
    num_nodes: int


class EventBatchDataset(Dataset):
    def __init__(self, events: TemporalEvents, start: int, end: int):
        self.events = events
        self.start = start
        self.end = end

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        i = self.start + idx
        return {
            "idx": i,
            "src": self.events.src[i],
            "dst": self.events.dst[i],
            "t": self.events.t[i],
            "y": self.events.y[i],
            "msg": self.events.msg[i],
        }


def collate_event_batch(batch):
    return {
        "idx": torch.tensor([x["idx"] for x in batch], dtype=torch.long),
        "src": torch.stack([x["src"] for x in batch]).long(),
        "dst": torch.stack([x["dst"] for x in batch]).long(),
        "t": torch.stack([x["t"] for x in batch]).float(),
        "y": torch.stack([x["y"] for x in batch]).float(),
        "msg": torch.stack([x["msg"] for x in batch]).float(),
    }


class TemporalEdgeDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        csv_path: str,
        batch_size: int = 1024,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_rows: Optional[int] = None,
        num_workers: int = 4,
        use_edge_features: bool = True,
        node_feature_mode: str = "degree",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.max_rows = max_rows
        self.num_workers = num_workers
        self.use_edge_features = use_edge_features
        self.node_feature_mode = node_feature_mode

        self.events: TemporalEvents | None = None
        self.train_start = 0
        self.train_end = 0
        self.val_end = 0

    def _make_node_features(self, df, num_nodes: int, train_end: int) -> torch.Tensor:
        if self.node_feature_mode == "constant":
            return torch.ones((num_nodes, 1), dtype=torch.float)

        if self.node_feature_mode == "degree":
            # Use TRAIN ONLY to avoid degree leakage from val/test.
            train_df = df.iloc[:train_end]
            in_deg = train_df.groupby("dst").size().reindex(range(num_nodes), fill_value=0).to_numpy(dtype=np.float32)
            out_deg = train_df.groupby("src").size().reindex(range(num_nodes), fill_value=0).to_numpy(dtype=np.float32)
            x_np = np.stack(
                [np.log1p(np.maximum(in_deg + out_deg, 0.0))],
                axis=1,
            ).astype(np.float32)
            return torch.tensor(x_np, dtype=torch.float)

        raise ValueError(f"Unknown node_feature_mode={self.node_feature_mode}")

    def setup(self, stage=None):
        df, num_nodes, train_end, val_end = load_unified_df(
            dataset_name=self.dataset_name,
            csv_path=self.csv_path,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            max_rows=self.max_rows,
        )

        extra_cols = []
        for c in df.columns:
            if c in {"src", "dst", "timestamp", "amount", "label"}:
                continue
            if df[c].dtype.kind in {"i", "u", "f", "b"}:
                extra_cols.append(c)

        msg_cols = ["amount", "timestamp"] + extra_cols if self.use_edge_features else []

        if len(msg_cols) == 0:
            msg = torch.zeros((len(df), 1), dtype=torch.float)
        else:
            msg = torch.tensor(df[msg_cols].to_numpy(dtype="float32"), dtype=torch.float)

        x = self._make_node_features(df, num_nodes=num_nodes, train_end=train_end)

        self.events = TemporalEvents(
            src=torch.tensor(df["src"].to_numpy(), dtype=torch.long),
            dst=torch.tensor(df["dst"].to_numpy(), dtype=torch.long),
            t=torch.tensor(df["timestamp"].to_numpy(dtype="float32"), dtype=torch.float),
            y=torch.tensor(df["label"].to_numpy(dtype="float32"), dtype=torch.float),
            msg=msg,
            x=x,
            num_nodes=num_nodes,
        )

        self.train_start = 0
        self.train_end = train_end
        self.val_end = val_end

        y = self.events.y.numpy()
        print(f"[Temporal] train={train_end}, val={val_end - train_end}, test={len(df) - val_end}")
        print(
            f"[Temporal] positive rates: "
            f"train={y[:train_end].mean():.6f}, "
            f"val={y[train_end:val_end].mean():.6f}, "
            f"test={y[val_end:].mean():.6f}"
        )
        print(f"[Temporal] msg_dim={self.events.msg.size(-1)}, node_feat_dim={self.events.x.size(-1)}")

    def train_dataloader(self):
        ds = EventBatchDataset(self.events, self.train_start, self.train_end)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_event_batch,
        )

    def val_dataloader(self):
        ds = EventBatchDataset(self.events, self.train_end, self.val_end)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_event_batch,
        )

    def test_dataloader(self):
        ds = EventBatchDataset(self.events, self.val_end, len(self.events.src))
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_event_batch,
        )