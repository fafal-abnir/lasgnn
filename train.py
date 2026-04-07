from __future__ import annotations

import argparse

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.data.datamodule_edge_minibatch import TransactionEdgeDataModule
from src.models.lightning_edge import LitEdgeClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["lasgnn", "gcn", "sage", "gat", "gin"])
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["amlsim", "samld", "saml-d", "bitcoin_alpha", "bitcoin_otc", "btc_alpha", "btc_otc", "alpha", "otc"],
    )
    parser.add_argument("--csv_path", type=str, required=True)

    parser.add_argument("--node_feature_mode", type=str, default="constant", choices=["constant", "degree"])

    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=[5, 5, 2])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pos_weight", type=float, default=100.0)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--no_lstm", action="store_true")
    parser.add_argument("--no_temporal_sort", action="store_true")
    parser.add_argument("--lstm_max_num_elements", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()
    L.seed_everything(42)

    dm = TransactionEdgeDataModule(
        dataset_name=args.dataset,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        num_neighbors=tuple(args.num_neighbors),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        temporal_sort=not args.no_temporal_sort,
        max_rows=args.max_rows,
        num_workers=args.num_workers,
        node_feature_mode=args.node_feature_mode,
    )
    dm.setup()

    model = LitEdgeClassifier(
        model_name=args.model,
        num_node_features=dm.train_graph.x.size(-1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        use_lstm=not args.no_lstm,
        dropout=args.dropout,
        lstm_max_num_elements=args.lstm_max_num_elements,
    )

    callbacks = [
        EarlyStopping(monitor="val_ap", mode="max", patience=20),
        ModelCheckpoint(monitor="val_ap", mode="max", save_top_k=1),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=CSVLogger("logs", name=f"{args.model}_{args.dataset}_{args.node_feature_mode}"),
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()