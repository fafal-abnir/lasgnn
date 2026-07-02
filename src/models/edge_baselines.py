from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


class BaseEdgeDecoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        h_src = h[src]
        h_dst = h[dst]
        z = torch.cat([h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst], dim=-1)
        return self.mlp(z)


class GCNEdge(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.decoder = BaseEdgeDecoder(hidden_dim, dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_label_index: Tensor):
        h = self.input_proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.decoder(h, edge_label_index), h


class SAGEEdge(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = dropout
        self.decoder = BaseEdgeDecoder(hidden_dim, dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_label_index: Tensor):
        h = self.input_proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.decoder(h, edge_label_index), h


class GATEdge(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList(
            [GATConv(hidden_dim, hidden_dim, heads=1, concat=False) for _ in range(num_layers)]
        )
        self.dropout = dropout
        self.decoder = BaseEdgeDecoder(hidden_dim, dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_label_index: Tensor):
        h = self.input_proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.decoder(h, edge_label_index), h


class GINEdge(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_node_features, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
        self.dropout = dropout
        self.decoder = BaseEdgeDecoder(hidden_dim, dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_label_index: Tensor):
        h = self.input_proj(x)
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.decoder(h, edge_label_index), h