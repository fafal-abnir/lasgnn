from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import BatchNorm


class FraudGTLayer(nn.Module):
    """
    FraudGT-style encoder block:
    - direct-neighbor attention
    - edge-based message passing gate
    - edge-based attention bias
    - node update
    - edge update
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if node_dim % num_heads != 0:
            raise ValueError("node_dim must be divisible by num_heads")

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)

        self.edge_gate_proj = nn.Linear(edge_dim, node_dim)
        self.edge_bias_proj = nn.Linear(edge_dim, num_heads)
        self.edge_out_proj = nn.Linear(num_heads, edge_dim)

        self.node_out_proj = nn.Linear(node_dim, node_dim)
        self.node_ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.edge_ffn = nn.Sequential(
            nn.Linear(edge_dim, edge_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim * 2, edge_dim),
        )

        self.node_norm1 = BatchNorm(node_dim)
        self.node_norm2 = BatchNorm(node_dim)
        self.edge_norm1 = BatchNorm(edge_dim)
        self.edge_norm2 = BatchNorm(edge_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)

        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)

        q_dst = q[dst]  # [E, H, D]
        k_src = k[src]  # [E, H, D]
        v_src = v[src]  # [E, H, D]

        # FraudGT-style edge-biased attention
        score_base = (q_dst * k_src).sum(dim=-1) / math.sqrt(self.head_dim)  # [E, H]
        edge_bias = self.edge_bias_proj(edge_attr)  # [E, H]
        attn_logits = score_base + edge_bias

        # Softmax over incoming edges of each destination, per head
        alpha = torch.zeros_like(attn_logits)
        for h in range(self.num_heads):
            logits_h = attn_logits[:, h]

            max_per_dst = torch.full(
                (num_nodes,),
                -1e30,
                device=logits_h.device,
                dtype=logits_h.dtype,
            )
            max_per_dst.scatter_reduce_(0, dst, logits_h, reduce="amax", include_self=True)

            logits_shifted = logits_h - max_per_dst[dst]
            exp_logits = torch.exp(logits_shifted)

            denom = torch.zeros(num_nodes, device=logits_h.device, dtype=logits_h.dtype)
            denom.scatter_add_(0, dst, exp_logits)

            alpha[:, h] = exp_logits / (denom[dst] + 1e-12)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        gate = torch.sigmoid(self.edge_gate_proj(edge_attr))  # [E, node_dim]
        gate = gate.view(num_edges, self.num_heads, self.head_dim)

        msg = alpha.unsqueeze(-1) * v_src * gate  # [E, H, D]

        out = torch.zeros(
            (num_nodes, self.num_heads, self.head_dim),
            device=x.device,
            dtype=x.dtype,
        )
        out.index_add_(0, dst, msg)
        out = out.reshape(num_nodes, self.node_dim)
        out = self.node_out_proj(out)

        x = self.node_norm1(x + out)
        x = self.node_norm2(x + self.node_ffn(x))

        edge_update = self.edge_out_proj(attn_logits)  # [E, edge_dim]
        edge_attr = self.edge_norm1(edge_attr + edge_update)
        edge_attr = self.edge_norm2(edge_attr + self.edge_ffn(edge_attr))

        return x, edge_attr


class FraudGTEdge(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_in = nn.Linear(num_node_features, hidden_dim)
        self.edge_in = nn.Linear(num_edge_features, edge_hidden_dim)

        self.layers = nn.ModuleList(
            [
                FraudGTLayer(
                    node_dim=hidden_dim,
                    edge_dim=edge_hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Eq. (9)-style edge classifier: h_i || E'_ij || h_j
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        x = self.node_in(x)
        edge_attr = self.edge_in(edge_attr)

        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        return x, edge_attr

    @staticmethod
    def _match_edge_features(
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_label_index: Tensor,
    ) -> Tensor:
        """
        Match each labeled edge (u,v) to an edge representation in the sampled subgraph.

        Since this pipeline does not pass explicit edge IDs, we resolve duplicates by
        taking the LAST matching occurrence of (u,v) in edge_index. This is much safer
        than taking the first match when the sampled graph is temporally sorted.
        """
        src_all, dst_all = edge_index
        src_lab, dst_lab = edge_label_index

        matched = []
        for u, v in zip(src_lab.tolist(), dst_lab.tolist()):
            mask = (src_all == u) & (dst_all == v)
            idx = torch.nonzero(mask, as_tuple=False).view(-1)

            if idx.numel() == 0:
                matched.append(
                    torch.zeros(
                        edge_attr.size(1),
                        device=edge_attr.device,
                        dtype=edge_attr.dtype,
                    )
                )
            else:
                matched.append(edge_attr[idx[-1]])

        return torch.stack(matched, dim=0)

    def decode(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_label_index: Tensor,
    ) -> Tensor:
        src_lab, dst_lab = edge_label_index

        e_lab = self._match_edge_features(
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_label_index=edge_label_index,
        )

        h_src = h[src_lab]
        h_dst = h[dst_lab]

        z = torch.cat([h_src, e_lab, h_dst], dim=-1)
        return self.edge_head(z)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_label_index: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h, e = self.encode(x, edge_index, edge_attr)
        logits = self.decode(h, edge_index, e, edge_label_index)
        return logits, (h, e)