from __future__ import annotations

import torch
from torch import nn

from src.temporal_models.temporal_common import EdgePredictor, HistoryBank, TimeEncoder


class MixerBlock(nn.Module):
    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Linear(seq_len, seq_len),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y

        y = self.norm2(x)
        y = self.channel_mlp(y)
        return x + y


class GraphMixerEdgeModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        msg_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        max_history: int = 50,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.hidden_dim = hidden_dim
        self.max_history = max_history

        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        self.time_enc = TimeEncoder(time_dim)
        self.token_proj = nn.Linear(hidden_dim + time_dim + 1 + msg_dim, hidden_dim)
        self.mixers = nn.ModuleList([MixerBlock(max_history, hidden_dim) for _ in range(num_layers)])
        self.readout = nn.Linear(hidden_dim, hidden_dim)
        self.edge_predictor = EdgePredictor(hidden_dim, msg_dim, hidden_dim)

    def encode_node(self, node_ids: torch.Tensor, times: torch.Tensor, history_bank: HistoryBank) -> torch.Tensor:
        device = node_ids.device
        out = []

        for nid, t in zip(node_ids.tolist(), times.tolist()):
            hist = history_bank.get_node_history(nid, t, self.max_history)
            if len(hist) == 0:
                out.append(self.node_emb(torch.tensor(nid, device=device)))
                continue

            nbr_ids = torch.tensor([x[0] for x in hist], dtype=torch.long, device=device)
            dt = torch.tensor([t - x[1] for x in hist], dtype=torch.float, device=device)
            sign = torch.tensor([x[2] for x in hist], dtype=torch.float, device=device).unsqueeze(-1)
            msg = torch.stack([x[3].to(device) for x in hist], dim=0)

            token = torch.cat([self.node_emb(nbr_ids), self.time_enc(dt), sign, msg], dim=-1)
            token = self.token_proj(token)

            if token.size(0) < self.max_history:
                pad = torch.zeros(self.max_history - token.size(0), token.size(1), device=device)
                token = torch.cat([pad, token], dim=0)
            else:
                token = token[-self.max_history:]

            token = token.unsqueeze(0)
            for mixer in self.mixers:
                token = mixer(token)
            out.append(self.readout(token[:, -1]).squeeze(0))

        return torch.stack(out, dim=0)

    def forward(self, batch, history_bank: HistoryBank):
        src = batch["src"]
        dst = batch["dst"]
        t = batch["t"]
        msg = batch["msg"]

        h_src = self.encode_node(src, t, history_bank)
        h_dst = self.encode_node(dst, t, history_bank)

        logits = self.edge_predictor(h_src, h_dst, msg)
        return logits