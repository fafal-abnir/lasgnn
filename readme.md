# TraceFormer Baselines

This repository contains the baseline implementations used to evaluate:

> **TraceFormer: A Role-Aware Temporal Trace Transformer for Money Laundering Detection**

## Baselines

- GRANDE
- LAS-GNN
- FraudGT
- Tab-AML
- TAML

## Evaluation Setting

All datasets are sorted chronologically and split into 70% training, 15% validation, and 15% testing.

### Graph-Based Baselines

Validation targets are evaluated on the training graph, while test targets are evaluated on the training-plus-validation graph.

For each prediction, the target transaction is excluded from the message-passing graph used to construct its own representation:

```text
Validation graph = Training transactions
Test graph       = Training + Validation transactions
```

### Tab-AML

Tab-AML does not use message passing. To reduce sender-receiver pair leakage, training transactions whose sender-receiver pair appears in the validation or test set are removed before training.

## Metrics

The primary metric is **AUCPR**. We also report AUROC, precision, recall, and F1 score.

Results are reported as mean and standard deviation over five runs using seeds `1, 2, 3, 4, 5`.

## Example Commands

Run commands from the repository root after placing the datasets in `raw_data/`.

### Bitcoin-OTC

```bash
python train.py --model lasgnn_edgefeat --dataset bitcoin_otc \
  --csv_path raw_data/soc-sign-bitcoinotc.csv \
  --lr 0.001 --max_epochs 100 --node_feature_mode degree \
  --pos_weight 10.0 --seed 3

python train.py --model multi_fraudgt --dataset bitcoin_otc \
  --csv_path raw_data/soc-sign-bitcoinotc.csv \
  --lr 0.001 --max_epochs 100 --node_feature_mode degree \
  --pos_weight 10.0 --seed 3

python train.py --model grande --dataset bitcoin_otc \
  --csv_path raw_data/soc-sign-bitcoinotc.csv \
  --lr 0.001 --max_epochs 50 \
  --pos_weight 10.0 --seed 3
```

### Bitcoin-Alpha

```bash
python train.py --model grande --dataset bitcoin_alpha \
  --csv_path raw_data/soc-sign-bitcoinalpha.csv \
  --lr 0.001 --max_epochs 50 \
  --pos_weight 10.0 --seed 3

python train.py --model lasgnn_edgefeat --dataset bitcoin_alpha \
  --csv_path raw_data/soc-sign-bitcoinalpha.csv \
  --lr 0.001 --max_epochs 100 --node_feature_mode degree \
  --pos_weight 10.0 --seed 3

python train.py --model multi_fraudgt --dataset bitcoin_alpha \
  --csv_path raw_data/soc-sign-bitcoinalpha.csv \
  --lr 0.001 --max_epochs 100 --node_feature_mode degree \
  --pos_weight 10.0 --seed 3
```

### AMLSim

```bash
python train.py --model lasgnn_edgefeat --dataset amlsim \
  --csv_path raw_data/aml_transactions.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 200.0 --seed 3

python train.py --model multi_fraudgt --dataset amlsim \
  --csv_path raw_data/aml_transactions.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 200.0 --seed 3

python train.py --model grande --dataset amlsim \
  --csv_path raw_data/aml_transactions.csv \
  --lr 0.001 --max_epochs 3 --node_feature_mode degree \
  --pos_weight 200.0 --seed 3
```

### SAML-D

```bash
python train.py --model lasgnn_edgefeat --dataset samld \
  --csv_path raw_data/SAML-D.csv \
  --lr 0.001 --max_epochs 100 --node_feature_mode degree \
  --pos_weight 10.0 --seed 3

python train.py --model multi_fraudgt --dataset samld \
  --csv_path raw_data/SAML-D.csv \
  --lr 0.001 --max_epochs 100 --node_feature_mode degree \
  --pos_weight 10.0 --seed 3
```

### AMLWorld-HI-Small

```bash
python train.py --model lasgnn_edgefeat --dataset amlworld_hi_small \
  --csv_path raw_data/HI-Small_Trans.csv \
  --lr 0.001 --max_epochs 100 --node_feature_mode degree \
  --pos_weight 10.0 --seed 1

python train.py --model multi_fraudgt --dataset amlworld_hi_small \
  --csv_path raw_data/HI-Small_Trans.csv \
  --lr 0.001 --max_epochs 100 --node_feature_mode degree \
  --pos_weight 10.0 --seed 1

python train.py --model taml --dataset amlworld_hi_small \
  --csv_path raw_data/HI-Small_Trans.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 200.0 --seed 1

python train_tabular.py --dataset amlworld_hi_small \
  --csv_path raw_data/HI-Small_Trans.csv \
  --train_ratio 0.75 --val_ratio 0.15 \
  --batch_size 2048 --hidden_dim 32 \
  --num_layers 4 --num_heads 4 --dropout 0.1 \
  --shared_embed_ratio 0.125 --seed 1
```

### AscendEXHacker

```bash
python train.py --model grande --dataset ascendexhacker \
  --csv_path raw_data/AscendEXHacker_transaction.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 200.0 --seed 1

python train.py --model lasgnn_edgefeat --dataset ascendexhacker \
  --csv_path raw_data/AscendEXHacker_transaction.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 200.0 --seed 1

python train.py --model multi_fraudgt --dataset ascendexhacker \
  --csv_path raw_data/AscendEXHacker_transaction.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 200.0 --seed 1

python train.py --model taml --dataset ascendexhacker \
  --csv_path raw_data/AscendEXHacker_transaction.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 200.0 --seed 1

python train_tabular.py --dataset ascendexhacker \
  --csv_path raw_data/AscendEXHacker_transaction.csv \
  --train_ratio 0.75 --val_ratio 0.15 \
  --batch_size 2048 --hidden_dim 32 \
  --num_layers 4 --num_heads 4 --dropout 0.1 \
  --shared_embed_ratio 0.125 --seed 1
```

### UpbitHack

```bash
python train.py --model grande --dataset upbithack \
  --csv_path raw_data/UpbitHack_transaction.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 10.0 --seed 1

python train.py --model lasgnn_edgefeat --dataset upbithack \
  --csv_path raw_data/UpbitHack_transaction.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 10.0 --seed 1

python train.py --model multi_fraudgt --dataset upbithack \
  --csv_path raw_data/UpbitHack_transaction.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 10.0 --seed 1

python train.py --model taml --dataset upbithack \
  --csv_path raw_data/UpbitHack_transaction.csv \
  --lr 0.001 --max_epochs 50 --node_feature_mode degree \
  --pos_weight 10.0 --seed 1

python train_tabular.py --dataset upbithack \
  --csv_path raw_data/UpbitHack_transaction.csv \
  --train_ratio 0.75 --val_ratio 0.15 \
  --batch_size 2048 --hidden_dim 32 \
  --num_layers 4 --num_heads 4 --dropout 0.1 \
  --shared_embed_ratio 0.125 --seed 1
```