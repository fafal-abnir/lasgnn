from __future__ import annotations

import random

import torch
from torch import Tensor, nn

from src.temporal_models.temporal_common import HistoryBank, TimeEncoder


class EdgePredictor(nn.Module):
    def __init__(self, emb_dim: int, msg_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4 + msg_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_src: Tensor, h_dst: Tensor, msg: Tensor) -> Tensor:
        z = torch.cat([h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst, msg], dim=-1)
        return self.mlp(z)


class CAWEdgeModel(nn.Module):
    """
    CAW-style edge classifier.

    It samples causal anonymous walks from past history:
      - no future edges
      - anonymous role features instead of raw node IDs inside walks
      - walk encoder via GRU
    """

    def __init__(
        self,
        num_nodes: int,
        msg_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        walk_len: int = 3,
        num_walks: int = 8,
        max_history: int = 20,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
        self.walk_len = walk_len
        self.num_walks = num_walks
        self.max_history = max_history

        self.time_enc = TimeEncoder(time_dim)

        # time encoding + sign + anonymous role features + msg
        self.step_dim = time_dim + 3 + msg_dim

        self.step_proj = nn.Linear(self.step_dim, hidden_dim)
        self.walk_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, hidden_dim)

        self.empty_node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.edge_predictor = EdgePredictor(hidden_dim, msg_dim, hidden_dim)

    def _sample_walk(self, start_node: int, cutoff_time: float, history_bank: HistoryBank):
        walk = []
        cur = start_node
        cur_t = cutoff_time
        visited = {start_node}

        for _ in range(self.walk_len):
            hist = history_bank.get_node_history(cur, cur_t, self.max_history)
            if len(hist) == 0:
                break

            nbr, tt, sign, msg = random.choice(hist)

            is_start = 1.0 if nbr == start_node else 0.0
            is_repeat = 1.0 if nbr in visited else 0.0
            is_new = 1.0 if nbr not in visited else 0.0

            walk.append((nbr, tt, sign, msg, is_start, is_repeat, is_new))

            visited.add(nbr)
            cur = nbr
            cur_t = tt

        return walk

    def encode_node(self, node_ids: Tensor, times: Tensor, history_bank: HistoryBank) -> Tensor:
        device = node_ids.device
        outs = []

        for nid, t in zip(node_ids.tolist(), times.tolist()):
            walk_tensors = []

            for _ in range(self.num_walks):
                walk = self._sample_walk(nid, t, history_bank)

                steps = []
                for (_nbr, tt, sign, msg, is_start, is_repeat, is_new) in walk:
                    dt = torch.tensor([t - tt], dtype=torch.float, device=device)
                    time_feat = self.time_enc(dt).squeeze(0)

                    role_feat = torch.tensor(
                        [sign, is_start + is_repeat, is_new],
                        dtype=torch.float,
                        device=device,
                    )

                    steps.append(torch.cat([time_feat, role_feat, msg.to(device)], dim=-1))

                if len(steps) == 0:
                    steps = [torch.zeros(self.step_dim, device=device)]

                walk_tensors.append(torch.stack(steps, dim=0))

            max_len = max(w.size(0) for w in walk_tensors)
            padded = []

            for w in walk_tensors:
                if w.size(0) < max_len:
                    pad = torch.zeros(max_len - w.size(0), w.size(1), device=device)
                    w = torch.cat([w, pad], dim=0)
                padded.append(w)

            x = torch.stack(padded, dim=0)  # [num_walks, L, step_dim]
            x = self.step_proj(x)

            _, h = self.walk_rnn(x)
            h = h.squeeze(0).mean(dim=0)

            # If no real history, add learned fallback embedding.
            if max_len == 1 and torch.all(padded[0] == 0):
                h = h + self.empty_node_emb(torch.tensor(nid, device=device))

            outs.append(self.readout(h))

        return torch.stack(outs, dim=0)

    def forward(self, batch: dict[str, Tensor], history_bank: HistoryBank) -> Tensor:
        src = batch["src"]
        dst = batch["dst"]
        t = batch["t"]
        msg = batch["msg"]

        h_src = self.encode_node(src, t, history_bank)
        h_dst = self.encode_node(dst, t, history_bank)

        return self.edge_predictor(h_src, h_dst, msg)