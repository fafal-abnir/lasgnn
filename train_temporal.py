from __future__ import annotations

import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.data.temporal_event_datamodule import TemporalEdgeDataModule
from src.temporal_models.lightning_temporal import LitTemporalEdgeModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True, choices=["tgn", "caw"])
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "amlsim",
            "samld",
            "saml-d",
            "bitcoin_alpha",
            "bitcoin_otc",
            "btc_alpha",
            "btc_otc",
            "alpha",
            "otc",
        ],
    )
    parser.add_argument("--csv_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--max_history", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pos_weight", type=float, default=10.0)
    parser.add_argument("--max_epochs", type=int, default=20)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--no_edge_features", action="store_true")
    parser.add_argument("--node_feature_mode", type=str, default="degree", choices=["degree", "constant"])

    return parser.parse_args()


def main():
    args = parse_args()
    L.seed_everything(42)

    dm = TemporalEdgeDataModule(
        dataset_name=args.dataset,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_rows=args.max_rows,
        num_workers=args.num_workers,
        use_edge_features=not args.no_edge_features,
        node_feature_mode=args.node_feature_mode,
    )
    dm.setup()

    model = LitTemporalEdgeModel(
        model_name=args.model,
        num_nodes=dm.events.num_nodes,
        msg_dim=dm.events.msg.size(-1),
        node_feat_dim=dm.events.x.size(-1),
        node_features=dm.events.x,
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        max_history=args.max_history,
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=5),
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=CSVLogger("logs", name=f"{args.model}_{args.dataset}_temporal"),
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()