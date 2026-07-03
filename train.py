from __future__ import annotations

import argparse
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.data.datamodule_edge_minibatch import TransactionEdgeDataModule
from src.models.lightning_edge import LitEdgeClassifier


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "lasgnn",
            "lasgnn_edgefeat",
            "gcn",
            "sage",
            "gat",
            "gin",
            "fraudgt",
            "fraudgt_rmp",
            "fraudgt_ports",
            "fraudgt_ego",
            "pe_fraudgt",
            "multi_fraudgt",
            "taml",
            "grande",
            "grande_reduced",
            "grande_no_time",
            "grande_no_cross",
            "grande_no_pruning",
            "grande_line",
        ],
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "amlsim",
            "samld",
            "saml-d",
            "amlworld",
            "amlworld_hi_small",
            "amlworld-hi-small",
            "amlworld_small_hi",
            "amlworld-small-hi",
            "hi-small",
            "bitcoin_alpha",
            "bitcoin_otc",
            "btc_alpha",
            "btc_otc",
            "alpha",
            "otc",
        ],
    )

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument(
        "--node_feature_mode",
        type=str,
        default="constant",
        choices=["constant", "degree", "enriched"],
    )

    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_neighbors", type=int, nargs="+", default=[10, 10])

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--edge_hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=1.0)

    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--no_lstm", action="store_true")
    parser.add_argument("--no_temporal_sort", action="store_true")
    parser.add_argument("--lstm_max_num_elements", type=int, default=16)

    parser.add_argument("--grande_time_dim", type=int, default=32)
    parser.add_argument("--grande_max_dual_neighbors", type=int, default=32)
    parser.add_argument("--grande_max_cross_neighbors", type=int, default=128)

    parser.add_argument("--taml_translation_dim", type=int, default=32)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.model == "taml" and args.node_feature_mode != "enriched":
        print(
            f"[TAML] Overriding node_feature_mode="
            f"'{args.node_feature_mode}' -> 'enriched'."
        )
        args.node_feature_mode = "enriched"

    if args.model == "taml" and args.pos_weight != 1.0:
        print(
            f"[TAML] Overriding pos_weight={args.pos_weight} -> 1.0 "
            f"(TAML trains on balanced undersampled batches)."
        )
        args.pos_weight = 1.0

    if args.model == "taml" and args.lr != 1e-3:
        print(f"[TAML] Overriding lr={args.lr} -> 0.001 (paper recipe).")
        args.lr = 1e-3

    if args.model == "taml" and args.weight_decay != 5e-3:
        print(
            f"[TAML] Overriding weight_decay={args.weight_decay} -> 0.005 "
            f"(paper recipe)."
        )
        args.weight_decay = 5e-3

    torch.set_float32_matmul_precision("high")
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
        model_name=args.model,
    )
    dm.setup()

    if args.model == "lasgnn_edgefeat":
        num_edge_features = dm.train_edge_label_attr.size(-1)
    else:
        num_edge_features = dm.train_graph.edge_attr.size(-1)

    num_target_edge_features = None
    if args.model == "taml":
        num_target_edge_features = dm.train_edge_label_attr.size(-1)

    model = LitEdgeClassifier(
        model_name=args.model,
        num_node_features=dm.train_graph.x.size(-1),
        num_edge_features=num_edge_features,
        hidden_dim=args.hidden_dim,
        edge_hidden_dim=args.edge_hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        use_lstm=not args.no_lstm,
        dropout=args.dropout,
        lstm_max_num_elements=args.lstm_max_num_elements,
        grande_time_dim=args.grande_time_dim,
        grande_max_dual_neighbors=args.grande_max_dual_neighbors,
        grande_max_cross_neighbors=args.grande_max_cross_neighbors,
        taml_translation_dim=args.taml_translation_dim,
        num_target_edge_features=num_target_edge_features,
    )

    if args.model == "taml":
        callbacks = [
            EarlyStopping(
                monitor="train_loss",
                mode="min",
                patience=16,
                min_delta=1e-2,
                check_on_train_epoch_end=True,
            ),
            ModelCheckpoint(
                monitor="val_ap",
                mode="max",
                save_top_k=1,
            ),
        ]
    else:
        callbacks = [
            EarlyStopping(
                monitor="val_ap",
                mode="max",
                patience=10,
            ),
            ModelCheckpoint(
                monitor="val_ap",
                mode="max",
                save_top_k=1,
            ),
        ]

    lightning_root_dir = "experiments"
    experiment_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiments_dir = f"{lightning_root_dir}/{args.model}/{args.dataset}/{experiment_datetime}"

    csv_logger = CSVLogger(
        save_dir=experiments_dir,
        version="",
    )
    csv_logger.log_hyperparams(vars(args))

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=csv_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        gradient_clip_val=1.0 if args.model == "taml" else None,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()