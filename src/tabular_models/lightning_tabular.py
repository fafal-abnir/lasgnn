from __future__ import annotations

import numpy as np
import lightning as L
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from src.tabular_models.tabaml import TabAML


class LitTabAML(L.LightningModule):
    """
    Lightning wrapper for TabAML.

    Validation and test metrics are computed over the complete epoch,
    not averaged from per-batch metrics.
    """

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

        if num_numeric < 0:
            raise ValueError("num_numeric cannot be negative.")

        if len(cat_cardinalities) != len(cat_feature_names):
            raise ValueError(
                "cat_cardinalities and cat_feature_names must have equal length."
            )

        if hidden_dim <= 0 or num_layers <= 0 or num_heads <= 0:
            raise ValueError(
                "hidden_dim, num_layers, and num_heads must be positive."
            )

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by "
                f"num_heads={num_heads}."
            )

        if pos_weight <= 0:
            raise ValueError("pos_weight must be positive.")

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

        self.register_buffer(
            "_pos_weight",
            torch.tensor(float(pos_weight), dtype=torch.float32),
        )

        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []
        self._test_logits: list[torch.Tensor] = []
        self._test_targets: list[torch.Tensor] = []

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.model(
            batch["x_num"],
            batch["x_cat"],
        )

    @staticmethod
    def _compute_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, float]:
        probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
            .reshape(-1)
        )

        labels = (
            targets
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.int64)
        )

        predictions = (
            probabilities >= 0.5
        ).astype(np.int64)

        metrics = {
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "auroc": float("nan"),
            "ap": float("nan"),
        }

        if np.unique(labels).size > 1:
            metrics["auroc"] = float(
                roc_auc_score(labels, probabilities)
            )
            metrics["ap"] = float(
                average_precision_score(labels, probabilities)
            )

        return metrics

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        stage: str,
    ) -> torch.Tensor:
        logits = self(batch).reshape(-1)
        targets = batch["y"].float().reshape(-1)

        if logits.shape != targets.shape:
            raise RuntimeError(
                f"Shape mismatch: logits={tuple(logits.shape)}, "
                f"targets={tuple(targets.shape)}"
            )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self._pos_weight,
        )

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=targets.numel(),
        )

        if stage == "val":
            self._val_logits.append(logits.detach().cpu())
            self._val_targets.append(targets.detach().cpu())
        elif stage == "test":
            self._test_logits.append(logits.detach().cpu())
            self._test_targets.append(targets.detach().cpu())

        return loss

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def on_validation_epoch_start(self) -> None:
        self._val_logits.clear()
        self._val_targets.clear()

    def on_test_epoch_start(self) -> None:
        self._test_logits.clear()
        self._test_targets.clear()

    def on_validation_epoch_end(self) -> None:
        if not self._val_logits:
            return

        logits = torch.cat(self._val_logits)
        targets = torch.cat(self._val_targets)
        metrics = self._compute_metrics(logits, targets)

        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(
                    f"val_{name}",
                    value,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                )

        self._val_logits.clear()
        self._val_targets.clear()

    def on_test_epoch_end(self) -> None:
        if not self._test_logits:
            return

        logits = torch.cat(self._test_logits)
        targets = torch.cat(self._test_targets)
        metrics = self._compute_metrics(logits, targets)

        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(
                    f"test_{name}",
                    value,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                )

        self._test_logits.clear()
        self._test_targets.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )