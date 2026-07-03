from __future__ import annotations

import torch
from torch import Tensor, linalg as LA, nn

from src.models.fraudgt_edge import FraudGTLayer


class TAMLEdge(nn.Module):
    """
    Edge-level TAML: FraudGT encoder + TAML translation unit.
    sigmoid(||h_VTN - h_dst|| - ||h_src + h_edge - h_dst||) is the
    laundering probability of transaction (src, dst).
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        edge_hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 128,
        translation_dim: int = 32,
        num_target_edge_features: int | None = None,
    ):
        super().__init__()
        if num_target_edge_features is None:
            num_target_edge_features = num_edge_features

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

        self.proj_translation = nn.Sequential(
            nn.Linear(hidden_dim, translation_dim),
            nn.LayerNorm(translation_dim),
            nn.LeakyReLU(),
        )

        self.mlp_translation = nn.Sequential(
            nn.Linear(num_target_edge_features, mlp_hidden_dim),
            nn.LayerNorm(mlp_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(mlp_hidden_dim, translation_dim),
            nn.LayerNorm(translation_dim),
            nn.LeakyReLU(),
        )

        self.h_VTN_translation = nn.Parameter(torch.randn(translation_dim))

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> tuple[Tensor, Tensor]:
        x = self.node_in(x)
        edge_attr = self.edge_in(edge_attr)

        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        return x, edge_attr

    def decode(
        self,
        h: Tensor,
        edge_label_index: Tensor,
        edge_label_attr: Tensor,
    ) -> Tensor:
        h_src = self.proj_translation(h[edge_label_index[0]])
        h_dst = self.proj_translation(h[edge_label_index[1]])
        h_edge = self.mlp_translation(edge_label_attr)

        logits = (
            LA.norm(self.h_VTN_translation - h_dst, dim=1)
            - LA.norm(h_src + h_edge - h_dst, dim=1)
        )
        return logits.unsqueeze(-1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_label_index: Tensor,
        edge_label_attr: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h, e = self.encode(x, edge_index, edge_attr)
        logits = self.decode(h, edge_label_index, edge_label_attr)
        return logits, (h, e)
