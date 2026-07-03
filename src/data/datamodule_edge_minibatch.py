from __future__ import annotations

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader

from src.data.unified import (
    build_transaction_graph_no_edge_features,
    build_transaction_graph_with_edge_features,
    load_unified_df,
)


LASGNN_EDGEFEAT_MODELS = {
    "lasgnn_edgefeat",
}

FRAUDGT_MODELS = {
    "fraudgt",
    "fraudgt_rmp",
    "fraudgt_ports",
    "fraudgt_ego",
    "pe_fraudgt",
    "multi_fraudgt",
}

GRANDE_MODELS = {
    "grande",
    "grande_reduced",
    "grande_no_time",
    "grande_no_cross",
    "grande_no_pruning",
    "grande_line",
}

TAML_MODELS = {
    "taml",
}


def sort_edges_by_dst_then_time(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_time: torch.Tensor,
    edge_id: torch.Tensor | None = None,
):
    dst = edge_index[1].to(torch.float64)
    t = edge_time.to(torch.float64)

    if t.numel() == 0:
        if edge_id is None:
            return edge_index, edge_attr, edge_time
        return edge_index, edge_attr, edge_time, edge_id

    t_shifted = t - t.min()
    stride = t_shifted.max().item() + 1.0
    key = dst * stride + t_shifted
    perm = torch.argsort(key, stable=True)

    if edge_id is None:
        return edge_index[:, perm], edge_attr[perm], edge_time[perm]

    return edge_index[:, perm], edge_attr[perm], edge_time[perm], edge_id[perm]


def make_reverse_signed_graph(data: Data, temporal_sort: bool = True) -> Data:
    """
    For LAS-GNN, LAS-GNN-EdgeFeat, GCN, SAGE, GAT, and GIN.

    Raw transaction features are not used in message passing.
    The only message-passing edge feature is:
        +1 = original transaction direction
        -1 = reverse direction
    """
    src, dst = data.edge_index
    rev_edge_index = torch.stack([dst, src], dim=0)

    fwd_sign = torch.ones(data.edge_index.size(1), 1, dtype=torch.float)
    rev_sign = -torch.ones(rev_edge_index.size(1), 1, dtype=torch.float)

    mp_edge_index = torch.cat([data.edge_index, rev_edge_index], dim=1)
    mp_edge_attr = torch.cat([fwd_sign, rev_sign], dim=0)
    mp_edge_time = torch.cat([data.edge_time, data.edge_time], dim=0)

    if temporal_sort:
        mp_edge_index, mp_edge_attr, mp_edge_time = sort_edges_by_dst_then_time(
            mp_edge_index,
            mp_edge_attr,
            mp_edge_time,
        )

    return Data(
        x=data.x,
        edge_index=mp_edge_index,
        edge_attr=mp_edge_attr,
        edge_time=mp_edge_time,
        num_nodes=data.num_nodes,
    )


def _compute_port_number_features(edge_index: torch.Tensor) -> torch.Tensor:
    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    num_edges = edge_index.size(1)

    if num_edges == 0:
        return torch.empty((0, 2), dtype=torch.float)

    in_ports = np.zeros(num_edges, dtype=np.float32)
    out_ports = np.zeros(num_edges, dtype=np.float32)

    incoming_neighbors: dict[int, set[int]] = {}
    for u, v in zip(src, dst):
        incoming_neighbors.setdefault(int(v), set()).add(int(u))

    incoming_map = {}
    for v, nbrs in incoming_neighbors.items():
        ordered = {nbr: rank + 1 for rank, nbr in enumerate(sorted(nbrs))}
        incoming_map[v] = (ordered, max(len(ordered), 1))

    outgoing_neighbors: dict[int, set[int]] = {}
    for u, v in zip(src, dst):
        outgoing_neighbors.setdefault(int(u), set()).add(int(v))

    outgoing_map = {}
    for u, nbrs in outgoing_neighbors.items():
        ordered = {nbr: rank + 1 for rank, nbr in enumerate(sorted(nbrs))}
        outgoing_map[u] = (ordered, max(len(ordered), 1))

    for i, (u, v) in enumerate(zip(src, dst)):
        u = int(u)
        v = int(v)

        in_rank, in_denom = incoming_map[v]
        out_rank, out_denom = outgoing_map[u]

        in_ports[i] = in_rank[u] / float(in_denom)
        out_ports[i] = out_rank[v] / float(out_denom)

    return torch.tensor(
        np.stack([in_ports, out_ports], axis=1),
        dtype=torch.float,
    )


def _augment_node_features_with_ego(data: Data) -> Data:
    ego = torch.zeros((data.x.size(0), 1), dtype=data.x.dtype)
    data.x = torch.cat([data.x, ego], dim=-1)
    return data


def make_fraudgt_graph(
    data: Data,
    temporal_sort: bool = True,
    use_rmp: bool = False,
    use_ports: bool = False,
    use_ego: bool = False,
) -> Data:
    """
    FraudGT graph construction.

    FraudGT uses raw edge features in message passing.

    Variants:
        fraudgt       -> raw directed edge features
        fraudgt_rmp   -> add reverse edges with direction one-hot
        fraudgt_ports -> append port-number features
        fraudgt_ego   -> add query-specific ego node feature
        pe_fraudgt    -> ports + ego
        multi_fraudgt -> RMP + ports + ego
    """
    src, dst = data.edge_index

    edge_index = data.edge_index
    edge_attr = data.edge_attr
    edge_time = data.edge_time
    edge_id = getattr(data, "edge_id", None)

    if use_rmp:
        fwd_dir = torch.tensor(
            [[1.0, 0.0]],
            dtype=edge_attr.dtype,
        ).repeat(edge_attr.size(0), 1)

        edge_attr = torch.cat([edge_attr, fwd_dir], dim=1)

        rev_edge_index = torch.stack([dst, src], dim=0)
        rev_edge_attr = data.edge_attr.clone()

        rev_dir = torch.tensor(
            [[0.0, 1.0]],
            dtype=rev_edge_attr.dtype,
        ).repeat(rev_edge_attr.size(0), 1)

        rev_edge_attr = torch.cat([rev_edge_attr, rev_dir], dim=1)

        edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        edge_attr = torch.cat([edge_attr, rev_edge_attr], dim=0)
        edge_time = torch.cat([edge_time, data.edge_time], dim=0)

        if edge_id is not None:
            edge_id = torch.cat([edge_id, edge_id], dim=0)

    if use_ports:
        ports = _compute_port_number_features(edge_index).to(edge_attr.dtype)
        edge_attr = torch.cat([edge_attr, ports], dim=1)

    if temporal_sort:
        if edge_id is None:
            edge_index, edge_attr, edge_time = sort_edges_by_dst_then_time(
                edge_index,
                edge_attr,
                edge_time,
            )
        else:
            edge_index, edge_attr, edge_time, edge_id = sort_edges_by_dst_then_time(
                edge_index,
                edge_attr,
                edge_time,
                edge_id,
            )

    out = Data(
        x=data.x.clone(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_time=edge_time,
        num_nodes=data.num_nodes,
    )

    if edge_id is not None:
        out.edge_id = edge_id

    if use_ego:
        out = _augment_node_features_with_ego(out)

    return out


class FraudGTEgoTransform:
    def __init__(self, use_ego: bool = False):
        self.use_ego = use_ego

    def __call__(self, batch: Data) -> Data:
        if not self.use_ego:
            return batch

        batch.x[:, -1] = 0.0
        endpoints = torch.unique(batch.edge_label_index.view(-1))
        batch.x[endpoints, -1] = 1.0
        return batch


class EdgeLabelAttrTransform:
    """
    Adds target transaction features and target edge IDs.

    Used by:
        GRANDE
        LAS-GNN-EdgeFeat
    """

    def __init__(
        self,
        edge_label_attr: torch.Tensor,
        edge_label_id: torch.Tensor,
    ):
        self.edge_label_attr = edge_label_attr.detach().cpu()
        self.edge_label_id = edge_label_id.detach().cpu()

    def __call__(self, batch: Data) -> Data:
        if not hasattr(batch, "input_id"):
            raise RuntimeError(
                "LinkNeighborLoader batch does not contain input_id. "
                "Cannot map edge_label_attr and edge_label_id."
            )

        input_id = batch.input_id.detach().cpu()

        batch.edge_label_attr = self.edge_label_attr[input_id]
        batch.edge_label_id = self.edge_label_id[input_id]

        return batch


def _filter_batch_edges(batch: Data, keep: torch.Tensor) -> Data:
    keep = keep.to(batch.edge_index.device)

    batch.edge_index = batch.edge_index[:, keep]

    if hasattr(batch, "edge_attr") and batch.edge_attr is not None:
        batch.edge_attr = batch.edge_attr[keep]

    if hasattr(batch, "edge_time") and batch.edge_time is not None:
        batch.edge_time = batch.edge_time[keep]

    if hasattr(batch, "edge_id") and batch.edge_id is not None:
        batch.edge_id = batch.edge_id[keep]

    if hasattr(batch, "e_id") and batch.e_id is not None:
        batch.e_id = batch.e_id[keep]

    return batch


class GrandeFastLeakageSafeTransform:
    """
    GRANDE-specific batch transform.

    It does three things:

    1. Adds target edge features and target edge IDs.
    2. Removes exact target edges from the sampled message-passing batch.
    3. Caps the sampled edge count before GRANDE builds its expensive
       edge-edge / dual graph.

    This keeps GRANDE leakage-controlled while making batches much faster.
    """

    def __init__(
        self,
        edge_label_attr: torch.Tensor,
        edge_label_id: torch.Tensor,
        max_edges: int = 4096,
        mask_target_edges: bool = True,
        prefer_incident_and_recent: bool = True,
    ):
        self.base_transform = EdgeLabelAttrTransform(
            edge_label_attr=edge_label_attr,
            edge_label_id=edge_label_id,
        )
        self.max_edges = max_edges
        self.mask_target_edges = mask_target_edges
        self.prefer_incident_and_recent = prefer_incident_and_recent

    def __call__(self, batch: Data) -> Data:
        batch = self.base_transform(batch)

        if batch.edge_index.size(1) == 0:
            return batch

        if self.mask_target_edges:
            batch = self._remove_target_edges(batch)

        if self.max_edges is not None and self.max_edges > 0:
            batch = self._cap_edges(batch)

        return batch

    def _remove_target_edges(self, batch: Data) -> Data:
        edge_id = None

        if hasattr(batch, "edge_id"):
            edge_id = batch.edge_id
        elif hasattr(batch, "e_id"):
            edge_id = batch.e_id

        edge_label_id = getattr(batch, "edge_label_id", None)

        if edge_id is None or edge_label_id is None:
            return batch

        edge_id = edge_id.to(batch.edge_index.device).long()
        edge_label_id = edge_label_id.to(batch.edge_index.device).long()

        keep = ~torch.isin(edge_id, edge_label_id)

        if keep.sum().item() == 0:
            return batch

        return _filter_batch_edges(batch, keep)

    def _cap_edges(self, batch: Data) -> Data:
        num_edges = batch.edge_index.size(1)

        if num_edges <= self.max_edges:
            return batch

        if not self.prefer_incident_and_recent:
            keep_idx = torch.arange(
                self.max_edges,
                device=batch.edge_index.device,
                dtype=torch.long,
            )
            keep = torch.zeros(
                num_edges,
                device=batch.edge_index.device,
                dtype=torch.bool,
            )
            keep[keep_idx] = True
            return _filter_batch_edges(batch, keep)

        src_all, dst_all = batch.edge_index
        endpoints = torch.unique(batch.edge_label_index.view(-1)).to(batch.edge_index.device)

        incident = torch.isin(src_all, endpoints) | torch.isin(dst_all, endpoints)

        if hasattr(batch, "edge_time") and batch.edge_time is not None:
            score = batch.edge_time.to(batch.edge_index.device).float()
        else:
            score = torch.arange(
                num_edges,
                device=batch.edge_index.device,
                dtype=torch.float,
            )

        # Strongly prefer edges incident to the current query endpoints.
        bonus = score.abs().max() + 1.0
        score = score + incident.float() * bonus * 2.0

        keep_idx = torch.topk(score, k=self.max_edges, largest=True).indices

        # Keep original relative order after top-k selection.
        keep_idx = torch.sort(keep_idx).values

        keep = torch.zeros(
            num_edges,
            device=batch.edge_index.device,
            dtype=torch.bool,
        )
        keep[keep_idx] = True

        return _filter_batch_edges(batch, keep)


class BalancedEdgeBatchSampler(torch.utils.data.Sampler):
    """
    Undersamples the majority class to the minority count every epoch and
    yields interleaved 0/1 batches of positions into edge_label_index.
    """

    def __init__(self, edge_label: torch.Tensor, batch_size: int):
        self.batch_size = batch_size
        edge_label = edge_label.detach().cpu()
        self.pos_idx = (edge_label == 1).nonzero(as_tuple=False).view(-1)
        self.neg_idx = (edge_label == 0).nonzero(as_tuple=False).view(-1)

    def __iter__(self):
        n = min(self.pos_idx.numel(), self.neg_idx.numel())
        pos = self.pos_idx[torch.randperm(self.pos_idx.numel())][:n]
        neg = self.neg_idx[torch.randperm(self.neg_idx.numel())][:n]

        interleaved = torch.empty(2 * n, dtype=torch.long)
        interleaved[0::2] = neg
        interleaved[1::2] = pos

        for start in range(0, 2 * n - self.batch_size + 1, self.batch_size):
            yield interleaved[start : start + self.batch_size].tolist()

    def __len__(self):
        return (2 * min(self.pos_idx.numel(), self.neg_idx.numel())) // self.batch_size


class TransactionEdgeDataModule(LightningDataModule):
    """
    Fast practical datamodule with leakage control.

    For all models:
        train graph = train
        val graph   = train only
        test graph  = train + val only

    For GRANDE specifically:
        same leakage-safe split graphs,
        but use smaller neighbor sampling and cap sampled batch edges
        before the GRANDE dual edge graph is constructed.
    """

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
        model_name: str = "lasgnn",
        grande_num_neighbors: tuple[int, ...] | None = (5, 5),
        grande_max_batch_edges: int = 4096,
        grande_mask_target_edges: bool = True,
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_neighbors = list(num_neighbors)

        if grande_num_neighbors is None:
            self.grande_num_neighbors = None
        else:
            self.grande_num_neighbors = list(grande_num_neighbors)

        self.grande_max_batch_edges = grande_max_batch_edges
        self.grande_mask_target_edges = grande_mask_target_edges

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.temporal_sort = temporal_sort
        self.max_rows = max_rows
        self.num_workers = num_workers
        self.node_feature_mode = node_feature_mode
        self.model_name = model_name.lower()

        self.train_graph: Data | None = None
        self.val_graph: Data | None = None
        self.test_graph: Data | None = None

        self.train_edge_label_index: torch.Tensor | None = None
        self.val_edge_label_index: torch.Tensor | None = None
        self.test_edge_label_index: torch.Tensor | None = None

        self.train_edge_label: torch.Tensor | None = None
        self.val_edge_label: torch.Tensor | None = None
        self.test_edge_label: torch.Tensor | None = None

        self.train_edge_label_attr: torch.Tensor | None = None
        self.val_edge_label_attr: torch.Tensor | None = None
        self.test_edge_label_attr: torch.Tensor | None = None

        self.train_edge_label_id: torch.Tensor | None = None
        self.val_edge_label_id: torch.Tensor | None = None
        self.test_edge_label_id: torch.Tensor | None = None

        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

    def _is_lasgnn_edgefeat(self) -> bool:
        return self.model_name in LASGNN_EDGEFEAT_MODELS

    def _is_fraudgt_family(self) -> bool:
        return self.model_name in FRAUDGT_MODELS

    def _is_grande_family(self) -> bool:
        return self.model_name in GRANDE_MODELS

    def _is_taml_family(self) -> bool:
        return self.model_name in TAML_MODELS

    def _uses_raw_edge_features(self) -> bool:
        return self._is_fraudgt_family() or self._is_grande_family() or self._is_taml_family()

    def _fraudgt_variant_flags(self) -> dict[str, bool]:
        name = self.model_name

        return {
            "use_rmp": name in {"fraudgt_rmp", "multi_fraudgt"},
            "use_ports": name in {"fraudgt_ports", "pe_fraudgt", "multi_fraudgt"},
            "use_ego": name in {"fraudgt_ego", "pe_fraudgt", "multi_fraudgt"},
        }

    @staticmethod
    def _edge_label_index_from_df(df) -> torch.Tensor:
        return torch.tensor(
            df[["src", "dst"]].to_numpy().T,
            dtype=torch.long,
        )

    @staticmethod
    def _edge_label_from_df(df) -> torch.Tensor:
        return torch.tensor(
            df["label"].to_numpy(dtype=np.float32),
            dtype=torch.float,
        )

    def setup(self, stage=None):
        df, num_nodes, train_end, val_end = load_unified_df(
            dataset_name=self.dataset_name,
            csv_path=self.csv_path,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            max_rows=self.max_rows,
        )

        df = df.reset_index(drop=True)

        train_df = df.iloc[:train_end].copy()
        val_target_df = df.iloc[train_end:val_end].copy()
        test_target_df = df.iloc[val_end:].copy()

        # Leakage-safe histories for every model, including GRANDE.
        val_history_df = df.iloc[:train_end].copy()
        test_history_df = df.iloc[:val_end].copy()

        if self._uses_raw_edge_features():
            train_base = build_transaction_graph_with_edge_features(
                train_df,
                num_nodes,
                node_feature_mode=self.node_feature_mode,
            )
            val_base = build_transaction_graph_with_edge_features(
                val_history_df,
                num_nodes,
                node_feature_mode=self.node_feature_mode,
            )
            test_base = build_transaction_graph_with_edge_features(
                test_history_df,
                num_nodes,
                node_feature_mode=self.node_feature_mode,
            )

            # Reference graph only for current target transaction features.
            # It is not used as the validation/test message-passing graph.
            full_raw_base = build_transaction_graph_with_edge_features(
                df,
                num_nodes,
                node_feature_mode=self.node_feature_mode,
            )

            train_base.edge_id = torch.arange(0, train_end, dtype=torch.long)
            val_base.edge_id = torch.arange(0, train_end, dtype=torch.long)
            test_base.edge_id = torch.arange(0, val_end, dtype=torch.long)

            self.train_edge_label_index = train_base.edge_index
            self.train_edge_label = train_base.edge_label
            self.train_edge_label_attr = full_raw_base.edge_attr[:train_end].clone()
            self.train_edge_label_id = torch.arange(0, train_end, dtype=torch.long)

            self.val_edge_label_index = self._edge_label_index_from_df(val_target_df)
            self.val_edge_label = self._edge_label_from_df(val_target_df)
            self.val_edge_label_attr = full_raw_base.edge_attr[train_end:val_end].clone()
            self.val_edge_label_id = torch.arange(train_end, val_end, dtype=torch.long)

            self.test_edge_label_index = self._edge_label_index_from_df(test_target_df)
            self.test_edge_label = self._edge_label_from_df(test_target_df)
            self.test_edge_label_attr = full_raw_base.edge_attr[val_end:].clone()
            self.test_edge_label_id = torch.arange(val_end, len(df), dtype=torch.long)

            if self._is_fraudgt_family():
                flags = self._fraudgt_variant_flags()

                self.train_graph = make_fraudgt_graph(
                    train_base,
                    temporal_sort=self.temporal_sort,
                    **flags,
                )
                self.val_graph = make_fraudgt_graph(
                    val_base,
                    temporal_sort=self.temporal_sort,
                    **flags,
                )
                self.test_graph = make_fraudgt_graph(
                    test_base,
                    temporal_sort=self.temporal_sort,
                    **flags,
                )

                use_ego = flags["use_ego"]
                self.train_transform = FraudGTEgoTransform(use_ego=use_ego)
                self.val_transform = FraudGTEgoTransform(use_ego=use_ego)
                self.test_transform = FraudGTEgoTransform(use_ego=use_ego)

            elif self._is_taml_family():
                self.train_graph = make_fraudgt_graph(
                    train_base,
                    temporal_sort=self.temporal_sort,
                    use_rmp=True,
                )
                self.val_graph = make_fraudgt_graph(
                    val_base,
                    temporal_sort=self.temporal_sort,
                    use_rmp=True,
                )
                self.test_graph = make_fraudgt_graph(
                    test_base,
                    temporal_sort=self.temporal_sort,
                    use_rmp=True,
                )

                self.train_transform = EdgeLabelAttrTransform(
                    edge_label_attr=self.train_edge_label_attr,
                    edge_label_id=self.train_edge_label_id,
                )
                self.val_transform = EdgeLabelAttrTransform(
                    edge_label_attr=self.val_edge_label_attr,
                    edge_label_id=self.val_edge_label_id,
                )
                self.test_transform = EdgeLabelAttrTransform(
                    edge_label_attr=self.test_edge_label_attr,
                    edge_label_id=self.test_edge_label_id,
                )

            else:
                # GRANDE:
                # Leakage-safe histories, but faster sampled batches.
                self.train_graph = train_base
                self.val_graph = val_base
                self.test_graph = test_base

                if self.temporal_sort:
                    (
                        self.train_graph.edge_index,
                        self.train_graph.edge_attr,
                        self.train_graph.edge_time,
                        self.train_graph.edge_id,
                    ) = sort_edges_by_dst_then_time(
                        self.train_graph.edge_index,
                        self.train_graph.edge_attr,
                        self.train_graph.edge_time,
                        self.train_graph.edge_id,
                    )

                    (
                        self.val_graph.edge_index,
                        self.val_graph.edge_attr,
                        self.val_graph.edge_time,
                        self.val_graph.edge_id,
                    ) = sort_edges_by_dst_then_time(
                        self.val_graph.edge_index,
                        self.val_graph.edge_attr,
                        self.val_graph.edge_time,
                        self.val_graph.edge_id,
                    )

                    (
                        self.test_graph.edge_index,
                        self.test_graph.edge_attr,
                        self.test_graph.edge_time,
                        self.test_graph.edge_id,
                    ) = sort_edges_by_dst_then_time(
                        self.test_graph.edge_index,
                        self.test_graph.edge_attr,
                        self.test_graph.edge_time,
                        self.test_graph.edge_id,
                    )

                self.train_transform = GrandeFastLeakageSafeTransform(
                    edge_label_attr=self.train_edge_label_attr,
                    edge_label_id=self.train_edge_label_id,
                    max_edges=self.grande_max_batch_edges,
                    mask_target_edges=self.grande_mask_target_edges,
                )

                self.val_transform = GrandeFastLeakageSafeTransform(
                    edge_label_attr=self.val_edge_label_attr,
                    edge_label_id=self.val_edge_label_id,
                    max_edges=self.grande_max_batch_edges,
                    mask_target_edges=self.grande_mask_target_edges,
                )

                self.test_transform = GrandeFastLeakageSafeTransform(
                    edge_label_attr=self.test_edge_label_attr,
                    edge_label_id=self.test_edge_label_id,
                    max_edges=self.grande_max_batch_edges,
                    mask_target_edges=self.grande_mask_target_edges,
                )

        else:
            train_base = build_transaction_graph_no_edge_features(
                train_df,
                num_nodes,
                node_feature_mode=self.node_feature_mode,
            )
            val_base = build_transaction_graph_no_edge_features(
                val_history_df,
                num_nodes,
                node_feature_mode=self.node_feature_mode,
            )
            test_base = build_transaction_graph_no_edge_features(
                test_history_df,
                num_nodes,
                node_feature_mode=self.node_feature_mode,
            )

            self.train_graph = make_reverse_signed_graph(
                train_base,
                temporal_sort=self.temporal_sort,
            )
            self.val_graph = make_reverse_signed_graph(
                val_base,
                temporal_sort=self.temporal_sort,
            )
            self.test_graph = make_reverse_signed_graph(
                test_base,
                temporal_sort=self.temporal_sort,
            )

            self.train_edge_label_index = train_base.edge_index
            self.train_edge_label = train_base.edge_label

            self.val_edge_label_index = self._edge_label_index_from_df(val_target_df)
            self.val_edge_label = self._edge_label_from_df(val_target_df)

            self.test_edge_label_index = self._edge_label_index_from_df(test_target_df)
            self.test_edge_label = self._edge_label_from_df(test_target_df)

            if self._is_lasgnn_edgefeat():
                full_raw_base = build_transaction_graph_with_edge_features(
                    df,
                    num_nodes,
                    node_feature_mode=self.node_feature_mode,
                )

                self.train_edge_label_attr = full_raw_base.edge_attr[:train_end].clone()
                self.val_edge_label_attr = full_raw_base.edge_attr[train_end:val_end].clone()
                self.test_edge_label_attr = full_raw_base.edge_attr[val_end:].clone()

                self.train_edge_label_id = torch.arange(0, train_end, dtype=torch.long)
                self.val_edge_label_id = torch.arange(train_end, val_end, dtype=torch.long)
                self.test_edge_label_id = torch.arange(val_end, len(df), dtype=torch.long)

                self.train_transform = EdgeLabelAttrTransform(
                    edge_label_attr=self.train_edge_label_attr,
                    edge_label_id=self.train_edge_label_id,
                )
                self.val_transform = EdgeLabelAttrTransform(
                    edge_label_attr=self.val_edge_label_attr,
                    edge_label_id=self.val_edge_label_id,
                )
                self.test_transform = EdgeLabelAttrTransform(
                    edge_label_attr=self.test_edge_label_attr,
                    edge_label_id=self.test_edge_label_id,
                )

        if self.node_feature_mode == "enriched":
            x_min = self.train_graph.x.min(dim=0).values
            x_range = self.train_graph.x.max(dim=0).values - x_min
            x_range = torch.where(x_range == 0, torch.ones_like(x_range), x_range)
            for graph in (self.train_graph, self.val_graph, self.test_graph):
                graph.x = (graph.x - x_min) / x_range

        print(
            f"[DataModule] model={self.model_name}, "
            f"node_dim={self.train_graph.x.size(-1)}, "
            f"edge_dim={self.train_graph.edge_attr.size(-1)}, "
            f"train_targets={self.train_edge_label.numel()}, "
            f"val_targets={self.val_edge_label.numel()}, "
            f"test_targets={self.test_edge_label.numel()}"
        )

        if self._is_grande_family():
            print(
                "[GRANDE FastLeakageSafe] "
                "train_graph=train; "
                "val_graph=train only; "
                "test_graph=train+val only; "
                f"grande_num_neighbors={self.grande_num_neighbors}; "
                f"grande_max_batch_edges={self.grande_max_batch_edges}; "
                f"mask_target_edges={self.grande_mask_target_edges}."
            )
        else:
            print(
                "[StrictLeakageSafeGraph] "
                "train_graph=train; "
                "val_graph=train only; "
                "test_graph=train+val only; "
                "validation/test target edges are excluded from their own message-passing graphs."
            )

        if self._is_grande_family() or self._is_lasgnn_edgefeat() or self._is_taml_family():
            print(
                f"[TargetEdgeFeatures] target_edge_attr_dim="
                f"{self.train_edge_label_attr.size(-1)}; "
                "target transaction features are passed through edge_label_attr."
            )

    def _loader(
        self,
        data: Data,
        edge_label_index: torch.Tensor,
        edge_label: torch.Tensor,
        transform,
        shuffle: bool,
    ):
        if self._is_grande_family() and self.grande_num_neighbors is not None:
            num_neighbors = self.grande_num_neighbors
        else:
            num_neighbors = self.num_neighbors

        return LinkNeighborLoader(
            data=data,
            num_neighbors=num_neighbors,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            batch_size=self.batch_size,
            shuffle=shuffle,
            neg_sampling=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
            transform=transform,
        )

    def train_dataloader(self):
        if self._is_taml_family():
            return LinkNeighborLoader(
                data=self.train_graph,
                num_neighbors=self.num_neighbors,
                edge_label_index=self.train_edge_label_index,
                edge_label=self.train_edge_label,
                batch_sampler=BalancedEdgeBatchSampler(
                    self.train_edge_label,
                    self.batch_size,
                ),
                neg_sampling=None,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                pin_memory=True,
                transform=self.train_transform,
            )

        return self._loader(
            data=self.train_graph,
            edge_label_index=self.train_edge_label_index,
            edge_label=self.train_edge_label,
            transform=self.train_transform,
            shuffle=True,
        )

    def val_dataloader(self):
        return self._loader(
            data=self.val_graph,
            edge_label_index=self.val_edge_label_index,
            edge_label=self.val_edge_label,
            transform=self.val_transform,
            shuffle=False,
        )

    def test_dataloader(self):
        return self._loader(
            data=self.test_graph,
            edge_label_index=self.test_edge_label_index,
            edge_label=self.test_edge_label,
            transform=self.test_transform,
            shuffle=False,
        )