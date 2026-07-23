from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from src.data.tabular_datamodule import TabularAMLDataModule
from src.tabular_models.lightning_tabular import LitTabAML


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "amlsim",
            "aml-sim",
            "samld",
            "saml-d",
            "amlworld",
            "aml-world",
            "amlworld_hi_small",
            "amlworld-hi-small",
            "hi-small",
            "upbithack",
            "ascendexhacker",
        ],
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--pos_weight",
        type=float,
        default=None,
        help=(
            "Positive-class weight. By default it is calculated "
            "from the filtered training split only."
        ),
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--shared_embed_ratio",
        type=float,
        default=0.125,
    )
    parser.add_argument(
        "--mlp_hidden_mult",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max_categories_per_feature",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--min_category_frequency",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--pair_disjoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For AMLWorld, enforce split disjointness using the ordered tuple "
            "(From Bank, Account, To Bank, Account.1)."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    L.seed_everything(
        args.seed,
        workers=True,
    )

    print(f"\033[93m{vars(args)}\033[0m")

    datamodule = TabularAMLDataModule(
        dataset_name=args.dataset,
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_rows=args.max_rows,
        num_workers=args.num_workers,
        max_categories_per_feature=(
            args.max_categories_per_feature
        ),
        min_category_frequency=(
            args.min_category_frequency
        ),
        pair_disjoint=args.pair_disjoint,
    )

    # Required before model construction because feature dimensions and
    # categorical mappings are fitted from the training split.
    datamodule.setup(stage="fit")

    pos_weight = (
        datamodule.train_pos_weight
        if args.pos_weight is None
        else float(args.pos_weight)
    )

    model = LitTabAML(
        num_numeric=datamodule.num_numeric_features,
        cat_cardinalities=datamodule.cat_cardinalities,
        cat_feature_names=datamodule.cat_feature_names,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        shared_embed_ratio=args.shared_embed_ratio,
        mlp_hidden_mult=args.mlp_hidden_mult,
    )

    experiment_time = datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )

    logger = CSVLogger(
        save_dir="experiments",
        name=f"tabaml/{args.dataset}",
        version=experiment_time,
    )
    logger.log_hyperparams(vars(args))
    checkpoint_dir = Path(logger.log_dir) / "checkpoints"

    callbacks = [
        EarlyStopping(
            monitor="val_ap",
            mode="max",
            patience=10,
        ),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            monitor="val_ap",
            mode="max",
            save_top_k=1,
            save_last=True,
            save_weights_only=False,
            filename="{epoch:02d}-{val_ap:.6f}",
        ),
    ]

    trainer = L.Trainer(
        default_root_dir=logger.log_dir,
        max_epochs=args.max_epochs,
        accelerator=(
            "gpu"
            if torch.cuda.is_available()
            else "cpu"
        ),
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(
        model,
        datamodule=datamodule,
    )

    # Test only the checkpoint selected by validation AP.
    trainer.test(
        model,
        datamodule=datamodule,
        ckpt_path="best",
    )


if __name__ == "__main__":
    main()