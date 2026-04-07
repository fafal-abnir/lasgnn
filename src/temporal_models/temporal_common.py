from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn


@dataclass
class HistoryBank:
    src: List[int]
    dst: List[int]
    t: List[float]
    msg: List[Tensor]

    @classmethod
    def empty(cls):
        return cls(src=[], dst=[], t=[], msg=[])

    def append(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor):
        self.src.extend(src.detach().cpu().tolist())
        self.dst.extend(dst.detach().cpu().tolist())
        self.t.extend(t.detach().cpu().tolist())
        self.msg.extend([m.detach().cpu() for m in msg])

    def get_node_history(self, node_id: int, cutoff_time: float, max_len: int):
        out = []
        for s, d, tt, m in zip(self.src, self.dst, self.t, self.msg):
            if tt >= cutoff_time:
                break
            if s == node_id or d == node_id:
                nbr = d if s == node_id else s
                sign = 1.0 if s == node_id else -1.0
                out.append((nbr, tt, sign, m))
        if len(out) > max_len:
            out = out[-max_len:]
        return out


class TimeEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(1, dim)

    def forward(self, dt: Tensor) -> Tensor:
        return torch.cos(self.lin(dt.unsqueeze(-1)))


class EdgePredictor(nn.Module):
    def __init__(self, emb_dim: int, msg_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = emb_dim * 4 + msg_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h_src: Tensor, h_dst: Tensor, msg: Tensor) -> Tensor:
        z = torch.cat(
            [h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst, msg],
            dim=-1,
        )
        return self.mlp(z)