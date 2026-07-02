from __future__ import annotations

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.models.edge_baselines import GCNEdge, GATEdge, GINEdge, SAGEEdge
from src.models.fraudgt_edge import FraudGTEdge
from src.models.grande_edge import GRANDEEdge
from src.models.lasgnn_edge import LASGNNEdge


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


class LitEdgeClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        edge_hidden_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        use_lstm: bool = True,
        dropout: float = 0.0,
        lstm_max_num_elements: int = 16,
        grande_time_dim: int = 32,
        grande_max_dual_neighbors: int = 32,
        grande_max_cross_neighbors: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay

        name = model_name.lower()
        self.model_name = name

        if name == "lasgnn":
            self.model = LASGNNEdge(
                num_node_features=num_node_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                use_lstm=use_lstm,
                dropout=dropout,
                lstm_max_num_elements=lstm_max_num_elements,
                use_edge_features_in_decoder=False,
            )

        elif name == "lasgnn_edgefeat":
            self.model = LASGNNEdge(
                num_node_features=num_node_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                use_lstm=use_lstm,
                dropout=dropout,
                lstm_max_num_elements=lstm_max_num_elements,
                num_edge_features=num_edge_features,
                use_edge_features_in_decoder=True,
            )

        elif name == "gcn":
            self.model = GCNEdge(
                num_node_features,
                hidden_dim,
                num_layers,
                dropout,
            )

        elif name == "sage":
            self.model = SAGEEdge(
                num_node_features,
                hidden_dim,
                num_layers,
                dropout,
            )

        elif name == "gat":
            self.model = GATEdge(
                num_node_features,
                hidden_dim,
                num_layers,
                dropout,
            )

        elif name == "gin":
            self.model = GINEdge(
                num_node_features,
                hidden_dim,
                num_layers,
                dropout,
            )

        elif name in FRAUDGT_MODELS:
            self.model = FraudGTEdge(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_dim=hidden_dim,
                edge_hidden_dim=edge_hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )

        elif name in GRANDE_MODELS:
            self.model = GRANDEEdge(
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                time_dim=grande_time_dim,
                variant=name,
                max_dual_neighbors=grande_max_dual_neighbors,
                max_cross_neighbors=grande_max_cross_neighbors,
            )

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

        self.register_buffer(
            "_pos_weight",
            torch.tensor([pos_weight], dtype=torch.float),
        )

    def forward(self, batch):
        if self.model_name in LASGNN_EDGEFEAT_MODELS:
            return self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.edge_label_index,
                batch.edge_label_attr,
            )

        if self.model_name in GRANDE_MODELS:
            edge_time = getattr(batch, "edge_time", None)

            edge_id = None
            if hasattr(batch, "edge_id"):
                edge_id = batch.edge_id
            elif hasattr(batch, "e_id"):
                edge_id = batch.e_id

            edge_label_id = getattr(batch, "edge_label_id", None)

            return self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.edge_label_index,
                batch.edge_label_attr,
                edge_id,
                edge_label_id,
                edge_time,
            )

        if self.model_name in FRAUDGT_MODELS:
            return self.model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.edge_label_index,
            )

        return self.model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.edge_label_index,
        )

    def _compute_metrics(self, logits: torch.Tensor, y: torch.Tensor):
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        target = y.detach().cpu().numpy().ravel().astype(int)
        pred = (probs >= 0.5).astype(int)

        metrics = {
            "f1": np.nan,
            "auroc": np.nan,
            "ap": np.nan,
        }

        if len(np.unique(target)) > 1:
            metrics["f1"] = float(f1_score(target, pred))
            metrics["auroc"] = float(roc_auc_score(target, probs))
            metrics["ap"] = float(average_precision_score(target, probs))

        return metrics

    def _shared_step(self, batch, stage: str):
        logits, _ = self(batch)

        y = batch.edge_label.view(-1, 1).float()

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=self._pos_weight.to(logits.device),
        )

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=y.size(0),
        )

        metrics = self._compute_metrics(logits, y)

        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(
                    f"{stage}_{name}",
                    value,
                    prog_bar=(stage != "train"),
                    on_step=False,
                    on_epoch=True,
                    batch_size=y.size(0),
                )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )