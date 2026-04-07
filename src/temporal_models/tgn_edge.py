from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class TGNState:
    memory: Tensor
    last_update: Tensor


class TimeEncoder(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.lin = nn.Linear(1, time_dim)

    def forward(self, delta_t: Tensor) -> Tensor:
        # delta_t: [N]
        return torch.cos(self.lin(delta_t.view(-1, 1)))


class RawMessageFunction(nn.Module):
    def __init__(
        self,
        memory_dim: int,
        msg_dim: int,
        time_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(memory_dim * 2 + msg_dim + time_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        src_memory: Tensor,
        dst_memory: Tensor,
        edge_msg: Tensor,
        time_encoding: Tensor,
    ) -> Tensor:
        x = torch.cat([src_memory, dst_memory, edge_msg, time_encoding], dim=-1)
        return self.mlp(x)


class LastMessageAggregator(nn.Module):
    """
    TGN usually aggregates messages per node before memory update.
    This version keeps the last message for each node in the batch,
    which is a common practical choice.
    """

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

    def forward(self, memory: Tensor, node_ids: Tensor, messages: Tensor) -> Tensor:
        if node_ids.numel() == 0:
            return memory

        old_mem = memory[node_ids]
        new_mem = self.gru(messages, old_mem)
        memory = memory.clone()
        memory[node_ids] = new_mem
        return memory


class GraphAttentionEmbedding(nn.Module):
    """
    Simplified TGN embedding module:
    embeds source/destination nodes using current memory, destination memory,
    raw edge message, and time encoding.
    """

    def __init__(
        self,
        memory_dim: int,
        msg_dim: int,
        time_dim: int,
        emb_dim: int,
    ):
        super().__init__()
        self.time_encoder = TimeEncoder(time_dim)
        self.src_proj = nn.Sequential(
            nn.Linear(memory_dim * 2 + msg_dim + time_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.dst_proj = nn.Sequential(
            nn.Linear(memory_dim * 2 + msg_dim + time_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(
        self,
        src_memory: Tensor,
        dst_memory: Tensor,
        edge_msg: Tensor,
        delta_t_src: Tensor,
        delta_t_dst: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src_t = self.time_encoder(delta_t_src)
        dst_t = self.time_encoder(delta_t_dst)

        src_x = torch.cat([src_memory, dst_memory, edge_msg, src_t], dim=-1)
        dst_x = torch.cat([dst_memory, src_memory, edge_msg, dst_t], dim=-1)

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
    More faithful TGN-style edge classifier.

    Workflow for each batch:
    1. Read current memory for src/dst
    2. Compute embeddings for prediction
    3. Predict current edges
    4. Build raw messages from current interactions
    5. Aggregate messages per node
    6. Update node memory
    """

    def __init__(
        self,
        num_nodes: int,
        msg_dim: int,
        memory_dim: int = 128,
        time_dim: int = 32,
        embedding_dim: int = 128,
        message_dim: int = 128,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.message_dim = message_dim

        self.raw_msg_time_encoder = TimeEncoder(time_dim)
        self.raw_message_function = RawMessageFunction(
            memory_dim=memory_dim,
            msg_dim=msg_dim,
            time_dim=time_dim,
            out_dim=message_dim,
        )
        self.message_aggregator = LastMessageAggregator()
        self.memory_updater = MemoryUpdater(memory_dim=memory_dim, message_dim=message_dim)

        self.embedding_module = GraphAttentionEmbedding(
            memory_dim=memory_dim,
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

    def reset_state(self):
        self._memory.zero_()
        self._last_update.zero_()

    def get_state(self) -> TGNState:
        return TGNState(memory=self._memory, last_update=self._last_update)

    def set_state(self, state: TGNState):
        self._memory.copy_(state.memory)
        self._last_update.copy_(state.last_update)

    def detach_state(self):
        self._memory.detach_()
        self._last_update.detach_()

    def compute_temporal_embeddings(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        msg: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src_memory = self._memory[src]
        dst_memory = self._memory[dst]

        delta_t_src = t - self._last_update[src]
        delta_t_dst = t - self._last_update[dst]

        h_src, h_dst = self.embedding_module(
            src_memory=src_memory,
            dst_memory=dst_memory,
            edge_msg=msg,
            delta_t_src=delta_t_src,
            delta_t_dst=delta_t_dst,
        )
        return h_src, h_dst

    def predict_edges(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        msg: Tensor,
    ) -> Tensor:
        h_src, h_dst = self.compute_temporal_embeddings(src, dst, t, msg)
        logits = self.edge_predictor(h_src, h_dst, msg)
        return logits

    @torch.no_grad()
    def update_memory(
        self,
        src: Tensor,
        dst: Tensor,
        t: Tensor,
        msg: Tensor,
    ):
        src_memory = self._memory[src]
        dst_memory = self._memory[dst]

        delta_t_src = t - self._last_update[src]
        delta_t_dst = t - self._last_update[dst]

        src_time_enc = self.raw_msg_time_encoder(delta_t_src)
        dst_time_enc = self.raw_msg_time_encoder(delta_t_dst)

        # message for source node from current interaction
        raw_src_msg = self.raw_message_function(
            src_memory=src_memory,
            dst_memory=dst_memory,
            edge_msg=msg,
            time_encoding=src_time_enc,
        )

        # message for destination node from current interaction
        raw_dst_msg = self.raw_message_function(
            src_memory=dst_memory,
            dst_memory=src_memory,
            edge_msg=msg,
            time_encoding=dst_time_enc,
        )

        all_nodes = torch.cat([src, dst], dim=0)
        all_msgs = torch.cat([raw_src_msg, raw_dst_msg], dim=0)
        all_ts = torch.cat([t, t], dim=0)

        updated_nodes, aggregated_msgs, updated_ts = self.message_aggregator(
            node_ids=all_nodes,
            messages=all_msgs,
            timestamps=all_ts,
            num_nodes=self.num_nodes,
        )

        self._memory = self.memory_updater(self._memory, updated_nodes, aggregated_msgs)
        self._last_update[updated_nodes] = updated_ts

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        src = batch["src"]
        dst = batch["dst"]
        t = batch["t"]
        msg = batch["msg"]
        return self.predict_edges(src, dst, t, msg)