from __future__ import annotations

import torch
import numpy as np
from lightning.pytorch import LightningDataModule
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.data.unified import build_transaction_graph_no_edge_features, load_unified_df


def sort_edges_by_dst_then_time(
    edge_index: torch.Tensor,
    edge_sign: torch.Tensor,
    edge_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dst = edge_index[1].to(torch.float64)
    t = edge_time.to(torch.float64)

    if t.numel() == 0:
        return edge_index, edge_sign, edge_time

    t = t - t.min()
    stride = t.max().item() + 1.0
    key = dst * stride + t
    perm = torch.argsort(key, stable=True)
    return edge_index[:, perm], edge_sign[perm], edge_time[perm]


def make_reverse_signed_graph(data: Data, temporal_sort: bool = True) -> Data:
    src, dst = data.edge_index
    rev_edge_index = torch.stack([dst, src], dim=0)

    fwd_sign = torch.ones(data.edge_index.size(1), 1, dtype=torch.float)
    rev_sign = -torch.ones(rev_edge_index.size(1), 1, dtype=torch.float)

    mp_edge_index = torch.cat([data.edge_index, rev_edge_index], dim=1)
    mp_edge_sign = torch.cat([fwd_sign, rev_sign], dim=0)
    mp_edge_time = torch.cat([data.edge_time, data.edge_time], dim=0)

    if temporal_sort:
        mp_edge_index, mp_edge_sign, mp_edge_time = sort_edges_by_dst_then_time(
            mp_edge_index, mp_edge_sign, mp_edge_time
        )

    return Data(
        x=data.x,
        edge_index=mp_edge_index,
        edge_attr=mp_edge_sign,
        edge_time=mp_edge_time,
        num_nodes=data.num_nodes,
    )


class TransactionEdgeDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        csv_path: str,
        batch_size: int = 2048,
        num_neighbors: tuple[int, ...] = (15, 10, 5),
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        temporal_sort: bool = True,
        max_rows: int | None = None,
        num_workers: int = 4,
        node_feature_mode: str = "constant",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_neighbors = list(num_neighbors)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.temporal_sort = temporal_sort
        self.max_rows = max_rows
        self.num_workers = num_workers
        self.node_feature_mode = node_feature_mode

    def setup(self, stage=None):
        df, num_nodes, train_end, val_end = load_unified_df(
            dataset_name=self.dataset_name,
            csv_path=self.csv_path,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            max_rows=self.max_rows,
        )

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[:val_end].copy()
        test_df = df.copy()

        val_target_df = df.iloc[train_end:val_end].copy()
        test_target_df = df.iloc[val_end:].copy()

        train_base = build_transaction_graph_no_edge_features(
            train_df, num_nodes, node_feature_mode=self.node_feature_mode
        )
        val_base = build_transaction_graph_no_edge_features(
            val_df, num_nodes, node_feature_mode=self.node_feature_mode
        )
        test_base = build_transaction_graph_no_edge_features(
            test_df, num_nodes, node_feature_mode=self.node_feature_mode
        )

        self.train_graph = make_reverse_signed_graph(train_base, temporal_sort=self.temporal_sort)
        self.val_graph = make_reverse_signed_graph(val_base, temporal_sort=self.temporal_sort)
        self.test_graph = make_reverse_signed_graph(test_base, temporal_sort=self.temporal_sort)

        self.train_edge_label_index = train_base.edge_index
        self.train_edge_label = train_base.edge_label

        self.val_edge_label_index = torch.tensor(
            val_target_df[["src", "dst"]].to_numpy().T, dtype=torch.long
        )
        self.val_edge_label = torch.tensor(
            val_target_df["label"].to_numpy(dtype=np.float32), dtype=torch.float
        )

        self.test_edge_label_index = torch.tensor(
            test_target_df[["src", "dst"]].to_numpy().T, dtype=torch.long
        )
        self.test_edge_label = torch.tensor(
            test_target_df["label"].to_numpy(dtype=np.float32), dtype=torch.float
        )

    def _loader(self, data: Data, edge_label_index: torch.Tensor, edge_label: torch.Tensor, shuffle: bool):
        return LinkNeighborLoader(
            data=data,
            num_neighbors=self.num_neighbors,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            batch_size=self.batch_size,
            shuffle=shuffle,
            neg_sampling=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_graph, self.train_edge_label_index, self.train_edge_label, True)

    def val_dataloader(self):
        return self._loader(self.val_graph, self.val_edge_label_index, self.val_edge_label, False)

    def test_dataloader(self):
        return self._loader(self.test_graph, self.test_edge_label_index, self.test_edge_label, False)