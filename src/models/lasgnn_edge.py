from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import LSTM, ModuleList
from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size
from torch_geometric.utils import spmm


class LSTMAggregation(Aggregation):
    def __init__(self, in_channels: int, out_channels: int, max_num_elements: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_num_elements = max_num_elements
        self.lstm = LSTM(
            input_size=in_channels,
            hidden_size=out_channels,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    @disable_dynamic_shapes(required_args=["dim_size", "max_num_elements"])
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:
        if max_num_elements is None:
            max_num_elements = self.max_num_elements

        dense_x, _ = self.to_dense_batch(
            x,
            index=index,
            ptr=ptr,
            dim_size=dim_size,
            dim=dim,
            max_num_elements=max_num_elements,
        )
        out, _ = self.lstm(dense_x)
        return out[:, -1]


class SignedGraphConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, tuple[int, int]],
        out_channels: int,
        aggr: str = "add",
        bias: bool = True,
        lstm_max_num_elements: int = 16,
        **kwargs,
    ):
        if isinstance(in_channels, int):
            aggr_in = in_channels
        else:
            aggr_in = in_channels[0]

        if aggr == "lstm":
            super().__init__(
                aggr=LSTMAggregation(aggr_in, aggr_in, max_num_elements=lstm_max_num_elements),
                **kwargs,
            )
        else:
            super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.sign_lin = Linear(1, in_channels[0], bias=True)
        self.lin_rel = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_root = Linear(in_channels[1], out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.sign_lin.reset_parameters()
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: Tensor,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        sign_weight = self.sign_lin(edge_attr)
        out = self.propagate(edge_index, x=x, edge_weight=sign_weight, size=size)
        out = self.lin_rel(out)

        x_r = x[1]
        if x_r is not None:
            out = out + self.lin_root(x_r)
        return out

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight * x_j

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)


class LASGNNEdge(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        use_lstm: bool = True,
        dropout: float = 0.0,
        lstm_max_num_elements: int = 4,
    ):
        super().__init__()
        self.node_emb = Linear(num_node_features, hidden_dim)
        self.dropout = dropout

        self.convs = ModuleList()
        self.norms = ModuleList()

        aggrs = ["add"] + ["lstm"] * (num_layers - 1) if use_lstm else ["add"] * num_layers

        for aggr in aggrs:
            self.convs.append(
                SignedGraphConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggr=aggr,
                    lstm_max_num_elements=lstm_max_num_elements,
                )
            )
            self.norms.append(BatchNorm(hidden_dim))

        self.edge_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        x = self.node_emb(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = 0.5 * (x + h)
        return x

    def decode(self, h: Tensor, edge_label_index: Tensor) -> Tensor:
        src, dst = edge_label_index
        h_src = h[src]
        h_dst = h[dst]
        z = torch.cat([h_src, h_dst, torch.abs(h_src - h_dst), h_src * h_dst], dim=-1)
        return self.edge_head(z)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, edge_label_index: Tensor):
        h = self.encode(x, edge_index, edge_attr)
        logits = self.decode(h, edge_label_index)
        return logits, h