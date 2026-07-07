from __future__ import annotations

import argparse
from datetime import datetime
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.data.tabular_datamodule import TabularAMLDataModule
from src.tabular_models.lightning_tabular import LitTabAML


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["amlsim", "samld"])
    parser.add_argument("--csv_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pos_weight", type=float, default=20.0)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--train_ratio", type=float, default=0.75)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--shared_embed_ratio", type=float, default=0.125)
    parser.add_argument("--mlp_hidden_mult", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed)
    print(f"\033[93m{vars(args)}\033[0m")

    dm = TabularAMLDataModule(
        dataset_name=args.dataset,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_rows=args.max_rows,
        num_workers=args.num_workers,
    )
    dm.setup()

    model = LitTabAML(
        num_numeric=dm.num_numeric_features,
        cat_cardinalities=dm.cat_cardinalities,
        cat_feature_names=dm.cat_feature_names,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        shared_embed_ratio=args.shared_embed_ratio,
        mlp_hidden_mult=args.mlp_hidden_mult,
    )

    callbacks = [
        EarlyStopping(monitor="val_ap", mode="max", patience=5),
        ModelCheckpoint(monitor="val_ap", mode="max", save_top_k=1, save_weights_only=True),
    ]
    lightning_root_dir = "experiments"
    experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    experiments_dir = f"{lightning_root_dir}/tabaml/{args.dataset}/{experiment_datetime}"
    csv_logger = CSVLogger(experiments_dir, version="")
    csv_logger.log_hyperparams(vars(args))
    trainer = L.Trainer(
        default_root_dir= experiments_dir,
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger = csv_logger,
        # logger=CSVLogger("logs", name=f"tabaml_{args.dataset}"),
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()