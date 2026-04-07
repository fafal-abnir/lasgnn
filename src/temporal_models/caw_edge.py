from __future__ import annotations

import random

import torch
from torch import nn

from src.temporal_models.temporal_common import EdgePredictor, HistoryBank, TimeEncoder


class CAWEdgeModel(nn.Module):
    """
    Practical CAW-style model:
    - extracts recent causal anonymous walk summaries from history
    - uses anonymized roles rather than raw node IDs inside the walk summary
    - predicts edge label from src/dst walk encodings + current message
    """

    def __init__(
        self,
        num_nodes: int,
        msg_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        walk_len: int = 3,
        num_walks: int = 8,
        max_history: int = 50,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
        self.walk_len = walk_len
        self.num_walks = num_walks
        self.max_history = max_history

        self.time_enc = TimeEncoder(time_dim)
        self.step_proj = nn.Linear(time_dim + 1 + msg_dim + 2, hidden_dim)
        self.walk_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, hidden_dim)
        self.edge_predictor = EdgePredictor(hidden_dim, msg_dim, hidden_dim)

    def _sample_walk(self, start_node: int, cutoff_time: float, history_bank: HistoryBank):
        walk = []
        cur = start_node
        cur_t = cutoff_time

        for depth in range(self.walk_len):
            hist = history_bank.get_node_history(cur, cur_t, self.max_history)
            if len(hist) == 0:
                break
            choice = random.choice(hist)
            nbr, tt, sign, msg = choice

            # simple anonymous role features:
            # [is_start, is_repeat]
            is_start = 1.0 if nbr == start_node else 0.0
            is_repeat = 1.0 if any(step[0] == nbr for step in walk) else 0.0

            walk.append((nbr, tt, sign, msg, is_start, is_repeat))
            cur = nbr
            cur_t = tt

        return walk

    def encode_node(self, node_ids: torch.Tensor, times: torch.Tensor, history_bank: HistoryBank) -> torch.Tensor:
        device = node_ids.device
        outs = []

        for nid, t in zip(node_ids.tolist(), times.tolist()):
            walk_tokens = []
            for _ in range(self.num_walks):
                walk = self._sample_walk(nid, t, history_bank)
                step_feats = []
                for (_nbr, tt, sign, msg, is_start, is_repeat) in walk:
                    dt = torch.tensor([t - tt], dtype=torch.float, device=device)
                    time_feat = self.time_enc(dt).squeeze(0)
                    msg = msg.to(device)
                    extra = torch.tensor([sign, is_start + is_repeat], dtype=torch.float, device=device)
                    step_feats.append(torch.cat([time_feat, extra, msg], dim=-1))

                if len(step_feats) == 0:
                    step_feats = [torch.zeros(self.step_proj.in_features, device=device)]
                steps = torch.stack(step_feats, dim=0)
                walk_tokens.append(steps)

            max_len = max(x.size(0) for x in walk_tokens)
            padded = []
            for steps in walk_tokens:
                if steps.size(0) < max_len:
                    pad = torch.zeros(max_len - steps.size(0), steps.size(1), device=device)
                    steps = torch.cat([steps, pad], dim=0)
                padded.append(steps)

            x = torch.stack(padded, dim=0)          # [num_walks, L, D]
            x = self.step_proj(x)
            _, h = self.walk_rnn(x)                 # [1, num_walks, H]
            h = h.squeeze(0).mean(dim=0)            # [H]
            outs.append(self.readout(h))

        return torch.stack(outs, dim=0)

    def forward(self, batch, history_bank: HistoryBank):
        src = batch["src"]
        dst = batch["dst"]
        t = batch["t"]
        msg = batch["msg"]

        h_src = self.encode_node(src, t, history_bank)
        h_dst = self.encode_node(dst, t, history_bank)
        return self.edge_predictor(h_src, h_dst, msg)