from __future__ import annotations

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.temporal_models.caw_edge import CAWEdgeModel
from src.temporal_models.graphmixer_edge import GraphMixerEdgeModel
from src.temporal_models.temporal_common import HistoryBank
from src.temporal_models.tgat_edge import TGATEdgeModel
from src.temporal_models.tgn_edge import TGNEdgeModel


class LitTemporalEdgeModel(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_nodes: int,
        msg_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        max_history: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        name = model_name.lower()
        if name == "tgat":
            self.model = TGATEdgeModel(
                num_nodes=num_nodes,
                msg_dim=msg_dim,
                memory_dim=hidden_dim,
                max_history=max_history,
            )
        elif name == "tgn":
            self.model = TGNEdgeModel(
                num_nodes=num_nodes,
                msg_dim=msg_dim,
                memory_dim=hidden_dim,
                time_dim=32,
                embedding_dim=hidden_dim,
                message_dim=hidden_dim,
            )
        elif name == "graphmixer":
            self.model = GraphMixerEdgeModel(
                num_nodes=num_nodes,
                msg_dim=msg_dim,
                hidden_dim=hidden_dim,
                max_history=max_history,
            )
        elif name == "caw":
            self.model = CAWEdgeModel(
                num_nodes=num_nodes,
                msg_dim=msg_dim,
                hidden_dim=hidden_dim,
                max_history=max_history,
            )
        else:
            raise ValueError(f"Unknown temporal model: {model_name}")

        self.model_name = name
        self.history_bank = HistoryBank.empty()
        self.register_buffer("_pos_weight", torch.tensor([pos_weight], dtype=torch.float))

    def reset_temporal_state(self):
        self.history_bank = HistoryBank.empty()
        if self.model_name == "tgn":
            self.model.reset_state()

    @torch.no_grad()
    def ingest_batch_into_state(self, batch):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.model_name == "tgn":
            self.model.update_memory(
                src=batch["src"],
                dst=batch["dst"],
                t=batch["t"],
                msg=batch["msg"],
            )

        self.history_bank.append(batch["src"], batch["dst"], batch["t"], batch["msg"])

    @torch.no_grad()
    def warmup_state_from_loader(self, loader):
        for batch in loader:
            self.ingest_batch_into_state(batch)

    def on_train_epoch_start(self):
        self.reset_temporal_state()

    def on_validation_epoch_start(self):
        self.reset_temporal_state()
        self.warmup_state_from_loader(self.trainer.datamodule.train_dataloader())

    def on_test_epoch_start(self):
        self.reset_temporal_state()
        self.warmup_state_from_loader(self.trainer.datamodule.train_dataloader())
        self.warmup_state_from_loader(self.trainer.datamodule.val_dataloader())

    def _compute_metrics(self, logits: torch.Tensor, y: torch.Tensor):
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        target = y.detach().cpu().numpy().ravel().astype(int)
        pred = (probs >= 0.5).astype(int)

        metrics = {"f1": np.nan, "auroc": np.nan, "ap": np.nan}
        if len(np.unique(target)) > 1:
            metrics["f1"] = float(f1_score(target, pred))
            metrics["auroc"] = float(roc_auc_score(target, probs))
            metrics["ap"] = float(average_precision_score(target, probs))
        return metrics

    def _shared_step(self, batch, stage: str):
        batch = {k: v.to(self.device) for k, v in batch.items()}

        if self.model_name == "tgn":
            logits = self.model(batch)
        else:
            logits = self.model(batch, self.history_bank)

        y = batch["y"].view(-1, 1).float()

        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=self._pos_weight.to(logits.device),
        )

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=y.size(0))

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

        if self.model_name == "tgn":
            self.model.update_memory(
                src=batch["src"],
                dst=batch["dst"],
                t=batch["t"],
                msg=batch["msg"],
            )
            self.model.detach_state()

        self.history_bank.append(batch["src"], batch["dst"], batch["t"], batch["msg"])
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)