from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.temporal_models.temporal_common import TimeEncoder


@dataclass
class TGNState:
    memory: Tensor
    last_update: Tensor


class RawMessageFunction(nn.Module):
    def __init__(
        self,
        memory_dim: int,
        node_feat_dim: int,
        msg_dim: int,
        time_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(memory_dim * 2 + node_feat_dim * 2 + msg_dim + time_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        src_memory: Tensor,
        dst_memory: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        edge_msg: Tensor,
        time_encoding: Tensor,
    ) -> Tensor:
        x = torch.cat(
            [src_memory, dst_memory, src_feat, dst_feat, edge_msg, time_encoding],
            dim=-1,
        )
        return self.mlp(x)


class LastMessageAggregator(nn.Module):
    def forward(self, node_ids: Tensor, messages: Tensor, timestamps: Tensor, num_nodes: int):
        device = messages.device
        msg_dim = messages.size(-1)

        aggregated = torch.zeros((num_nodes, msg_dim), device=device, dtype=messages.dtype)
        has_msg = torch.zeros(num_nodes, device=device, dtype=torch.bool)
        last_ts = torch.full((num_nodes,), -1e30, device=device, dtype=timestamps.dtype)

        for i in range(node_ids.numel()):
            nid = int(node_ids[i].item())
            ts = timestamps[i]
            if ts >= last_ts[nid]:
                last_ts[nid] = ts
                aggregated[nid] = messages[i]
                has_msg[nid] = True

        updated_nodes = torch.nonzero(has_msg, as_tuple=False).view(-1)
        return updated_nodes, aggregated[updated_nodes], last_ts[updated_nodes]


class MemoryUpdater(nn.Module):
    def __init__(self, memory_dim: int, message_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(message_dim, memory_dim)

    def compute(self, old_memory: Tensor, messages: Tensor) -> Tensor:
        return self.gru(messages, old_memory)

    def update(self, memory: Tensor, node_ids: Tensor, messages: Tensor) -> Tensor:
        if node_ids.numel() == 0:
            return memory
        old_mem = memory[node_ids]
        new_mem = self.compute(old_mem, messages)
        memory = memory.clone()
        memory[node_ids] = new_mem
        return memory


class TemporalEmbeddingModule(nn.Module):
    def __init__(
        self,
        memory_dim: int,
        node_feat_dim: int,
        msg_dim: int,
        time_dim: int,
        emb_dim: int,
    ):
        super().__init__()
        self.time_encoder = TimeEncoder(time_dim)

        in_dim = memory_dim * 2 + node_feat_dim * 2 + msg_dim + time_dim
        self.src_proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.dst_proj = nn.Sequential(
            nn.Linear(in_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(
        self,
        src_memory: Tensor,
        dst_memory: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        edge_msg: Tensor,
        delta_t_src: Tensor,
        delta_t_dst: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src_t = self.time_encoder(delta_t_src)
        dst_t = self.time_encoder(delta_t_dst)

        src_x = torch.cat([src_memory, dst_memory, src_feat, dst_feat, edge_msg, src_t], dim=-1)
        dst_x = torch.cat([dst_memory, src_memory, dst_feat, src_feat, edge_msg, dst_t], dim=-1)

        h_src = self.src_proj(src_x)
        h_dst = self.dst_proj(dst_x)
        return h_src, h_dst


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


class TGNEdgeModel(nn.Module):
    """
    TGN-style edge classifier with node features.

    Prediction uses:
      - dynamic memory
      - static node features
      - edge/message features
      - time encoding

    Memory is updated after prediction in Lightning.
    """

    def __init__(
        self,
        num_nodes: int,
        node_feat_dim: int,
        msg_dim: int,
        memory_dim: int = 128,
        time_dim: int = 32,
        embedding_dim: int = 128,
        message_dim: int = 128,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.msg_dim = msg_dim
        self.memory_dim = memory_dim

        self.node_feat_proj = nn.Linear(node_feat_dim, memory_dim)
        self.raw_msg_time_encoder = TimeEncoder(time_dim)

        self.raw_message_function = RawMessageFunction(
            memory_dim=memory_dim,
            node_feat_dim=node_feat_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            out_dim=message_dim,
        )

        self.message_aggregator = LastMessageAggregator()
        self.memory_updater = MemoryUpdater(memory_dim=memory_dim, message_dim=message_dim)

        self.embedding_module = TemporalEmbeddingModule(
            memory_dim=memory_dim,
            node_feat_dim=node_feat_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            emb_dim=embedding_dim,
        )

        self.edge_predictor = EdgePredictor(
            emb_dim=embedding_dim,
            msg_dim=msg_dim,
            hidden_dim=embedding_dim,
        )

        self.register_buffer("_memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("_last_update", torch.zeros(num_nodes))
        self.register_buffer("_x", torch.zeros(num_nodes, node_feat_dim))

    def set_node_features(self, x: Tensor):
        if x.size(0) != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {x.size(0)}")
        if x.size(1) != self.node_feat_dim:
            raise ValueError(f"Expected node_feat_dim={self.node_feat_dim}, got {x.size(1)}")
        self._x = x.detach().to(self._x.device)

    def reset_state(self):
        self._memory.zero_()
        self._last_update.zero_()

    def detach_state(self):
        self._memory.detach_()
        self._last_update.detach_()

    def _effective_memory(self, node_ids: Tensor) -> Tensor:
        return self._memory[node_ids] + self.node_feat_proj(self._x[node_ids])

    def _raw_messages(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        msg: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src_memory = self._effective_memory(src)
        dst_memory = self._effective_memory(dst)

        src_feat = self._x[src]
        dst_feat = self._x[dst]

        delta_t_src = t - self._last_update[src]
        delta_t_dst = t - self._last_update[dst]

        src_time_enc = self.raw_msg_time_encoder(delta_t_src)
        dst_time_enc = self.raw_msg_time_encoder(delta_t_dst)

        raw_src_msg = self.raw_message_function(
            src_memory=src_memory,
            dst_memory=dst_memory,
            src_feat=src_feat,
            dst_feat=dst_feat,
            edge_msg=msg,
            time_encoding=src_time_enc,
        )

        raw_dst_msg = self.raw_message_function(
            src_memory=dst_memory,
            dst_memory=src_memory,
            src_feat=dst_feat,
            dst_feat=src_feat,
            edge_msg=msg,
            time_encoding=dst_time_enc,
        )

        return raw_src_msg, raw_dst_msg

    def predict_edges(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor) -> Tensor:
        src_memory = self._effective_memory(src)
        dst_memory = self._effective_memory(dst)

        src_feat = self._x[src]
        dst_feat = self._x[dst]

        delta_t_src = t - self._last_update[src]
        delta_t_dst = t - self._last_update[dst]

        # Prospective memory update for the current interaction.
        # This makes message/updater parameters trainable while actual state is still updated after prediction.
        raw_src_msg, raw_dst_msg = self._raw_messages(src, dst, t, msg)
        src_memory_prospective = self.memory_updater.compute(src_memory, raw_src_msg)
        dst_memory_prospective = self.memory_updater.compute(dst_memory, raw_dst_msg)

        h_src, h_dst = self.embedding_module(
            src_memory=src_memory_prospective,
            dst_memory=dst_memory_prospective,
            src_feat=src_feat,
            dst_feat=dst_feat,
            edge_msg=msg,
            delta_t_src=delta_t_src,
            delta_t_dst=delta_t_dst,
        )

        return self.edge_predictor(h_src, h_dst, msg)

    @torch.no_grad()
    def update_memory(self, src: Tensor, dst: Tensor, t: Tensor, msg: Tensor):
        raw_src_msg, raw_dst_msg = self._raw_messages(src, dst, t, msg)

        all_nodes = torch.cat([src, dst], dim=0)
        all_msgs = torch.cat([raw_src_msg, raw_dst_msg], dim=0)
        all_ts = torch.cat([t, t], dim=0)

        updated_nodes, aggregated_msgs, updated_ts = self.message_aggregator(
            node_ids=all_nodes,
            messages=all_msgs,
            timestamps=all_ts,
            num_nodes=self.num_nodes,
        )

        self._memory = self.memory_updater.update(self._memory, updated_nodes, aggregated_msgs)
        self._last_update[updated_nodes] = updated_ts

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        return self.predict_edges(
            src=batch["src"],
            dst=batch["dst"],
            t=batch["t"],
            msg=batch["msg"],
        )