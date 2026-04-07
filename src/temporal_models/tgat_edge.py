from __future__ import annotations

import torch
from torch import nn

from src.temporal_models.temporal_common import EdgePredictor, HistoryBank, TimeEncoder


class TGATEdgeModel(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        msg_dim: int,
        memory_dim: int = 128,
        time_dim: int = 32,
        max_history: int = 50,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.memory_dim = memory_dim
        self.max_history = max_history

        self.node_emb = nn.Embedding(num_nodes, memory_dim)
        self.time_enc = TimeEncoder(time_dim)
        self.hist_proj = nn.Linear(memory_dim + time_dim + 1 + msg_dim, memory_dim)
        self.attn = nn.MultiheadAttention(memory_dim, num_heads=4, batch_first=True)
        self.edge_predictor = EdgePredictor(memory_dim, msg_dim, memory_dim)

    def encode_node(self, node_ids: torch.Tensor, times: torch.Tensor, history_bank: HistoryBank) -> torch.Tensor:
        device = node_ids.device
        out = []

        for nid, t in zip(node_ids.tolist(), times.tolist()):
            hist = history_bank.get_node_history(nid, t, self.max_history)
            q = self.node_emb(torch.tensor([nid], device=device))

            if len(hist) == 0:
                out.append(q.squeeze(0))
                continue

            nbr_ids = torch.tensor([x[0] for x in hist], dtype=torch.long, device=device)
            dt = torch.tensor([t - x[1] for x in hist], dtype=torch.float, device=device)
            sign = torch.tensor([x[2] for x in hist], dtype=torch.float, device=device).unsqueeze(-1)
            msg = torch.stack([x[3].to(device) for x in hist], dim=0)

            nbr_emb = self.node_emb(nbr_ids)
            time_emb = self.time_enc(dt)
            tokens = torch.cat([nbr_emb, time_emb, sign, msg], dim=-1)
            tokens = self.hist_proj(tokens).unsqueeze(0)

            attn_out, _ = self.attn(q.unsqueeze(0), tokens, tokens)
            out.append(attn_out.squeeze(0).squeeze(0))

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