from __future__ import annotations

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.tabular_models.tabaml import TabAML


class LitTabAML(L.LightningModule):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: list[int],
        cat_feature_names: list[str],
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        pos_weight: float = 1.0,
        shared_embed_ratio: float = 0.125,
        mlp_hidden_mult: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = TabAML(
            num_numeric=num_numeric,
            cat_cardinalities=cat_cardinalities,
            cat_feature_names=cat_feature_names,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            shared_embed_ratio=shared_embed_ratio,
            mlp_hidden_mult=mlp_hidden_mult,
        )

        self.register_buffer("_pos_weight", torch.tensor([pos_weight], dtype=torch.float))

    def forward(self, batch):
        return self.model(batch["x_num"], batch["x_cat"])

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
        logits = self(batch)
        y = batch["y"]

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
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )