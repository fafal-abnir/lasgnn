from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.utils import softmax


class BochnerTimeEncoder(nn.Module):
    """
    GRANDE / TGAT-style functional time encoding:
        TE(s) = [cos(rho_i s), sin(rho_i s)]
    """

    def __init__(self, time_dim: int):
        super().__init__()
        if time_dim % 2 != 0:
            raise ValueError("time_dim must be even.")
        self.half_dim = time_dim // 2
        self.rho = nn.Parameter(torch.randn(self.half_dim) * 0.01)

    def forward(self, dt: Tensor) -> Tensor:
        dt = dt.view(-1, 1)
        phase = dt * self.rho.view(1, -1)
        out = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        return out / math.sqrt(self.half_dim)


class GRANDEAttentionBlock(nn.Module):
    """
    Transformer-style GRANDE attention:

        ATTN(h_v, {h_u, g_uv})
        = sum alpha_uv W_N h_u + beta_uv W_E g_uv

    This block is used for both incoming and outgoing node aggregation.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        time_dim: int = 0,
    ):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads.")

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.dropout = dropout
        self.time_dim = time_dim

        self.q_node = nn.Linear(node_dim, out_dim)
        self.k_node = nn.Linear(node_dim, out_dim)
        self.v_node = nn.Linear(node_dim, out_dim)

        self.k_edge = nn.Linear(edge_dim + time_dim, out_dim)
        self.v_edge = nn.Linear(edge_dim + time_dim, out_dim)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.res_proj = nn.Linear(node_dim, out_dim)

        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(
        self,
        h: Tensor,
        edge_index_for_agg: Tensor,
        g: Tensor,
        dim_size: int,
        edge_time_feat: Optional[Tensor] = None,
    ) -> Tensor:
        src, dst = edge_index_for_agg

        if edge_time_feat is not None:
            g_in = torch.cat([g, edge_time_feat], dim=-1)
        else:
            g_in = g

        q = self.q_node(h[dst]).view(-1, self.num_heads, self.head_dim)
        k_node = self.k_node(h[src]).view(-1, self.num_heads, self.head_dim)
        v_node = self.v_node(h[src]).view(-1, self.num_heads, self.head_dim)

        k_edge = self.k_edge(g_in).view(-1, self.num_heads, self.head_dim)
        v_edge = self.v_edge(g_in).view(-1, self.num_heads, self.head_dim)

        score_node = (q * k_node).sum(dim=-1) / math.sqrt(self.head_dim)
        score_edge = (q * k_edge).sum(dim=-1) / math.sqrt(self.head_dim)

        alpha = softmax(score_node, dst, num_nodes=dim_size)
        beta = softmax(score_edge, dst, num_nodes=dim_size)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        beta = F.dropout(beta, p=self.dropout, training=self.training)

        msg = alpha.unsqueeze(-1) * v_node + beta.unsqueeze(-1) * v_edge

        out = torch.zeros(
            dim_size,
            self.num_heads,
            self.head_dim,
            dtype=h.dtype,
            device=h.device,
        )
        out.index_add_(0, dst, msg)
        out = out.reshape(dim_size, self.out_dim)
        out = self.out_proj(out)

        h_res = self.res_proj(h)
        z = self.norm1(h_res + out)
        z = self.norm2(z + self.ffn(z))
        return z


class GRANDEDualAttentionBlock(nn.Module):
    """
    Transformer block over the augmented edge-adjacency graph.

    Original transaction edges become dual-graph nodes.
    Dual edges connect transactions sharing an endpoint.

    The context for edge-edge attention is:
        h_common_node + type_embedding(edge_edge_type)
    where edge_edge_type is one of:
        0 head-to-head
        1 head-to-tail
        2 tail-to-head
        3 tail-to-tail
    """

    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        time_dim: int = 0,
        num_edge_types: int = 4,
    ):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError("out_dim must be divisible by num_heads.")

        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.dropout = dropout
        self.time_dim = time_dim

        self.type_emb = nn.Embedding(num_edge_types, node_dim)

        self.q_edge = nn.Linear(edge_dim, out_dim)
        self.k_edge = nn.Linear(edge_dim, out_dim)
        self.v_edge = nn.Linear(edge_dim, out_dim)

        self.k_ctx = nn.Linear(node_dim + time_dim, out_dim)
        self.v_ctx = nn.Linear(node_dim + time_dim, out_dim)

        self.out_proj = nn.Linear(out_dim, out_dim)
        self.res_proj = nn.Linear(edge_dim, out_dim)

        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(
        self,
        g: Tensor,
        h: Tensor,
        dual_edge_index: Tensor,
        dual_edge_type: Tensor,
        dual_common_node: Tensor,
        num_edges: int,
        dual_time_feat: Optional[Tensor] = None,
    ) -> Tensor:
        if dual_edge_index.numel() == 0:
            z = self.res_proj(g)
            z = self.norm1(z)
            z = self.norm2(z + self.ffn(z))
            return z

        src_e, dst_e = dual_edge_index

        ctx = h[dual_common_node] + self.type_emb(dual_edge_type)

        if dual_time_feat is not None:
            ctx = torch.cat([ctx, dual_time_feat], dim=-1)

        q = self.q_edge(g[dst_e]).view(-1, self.num_heads, self.head_dim)

        k_edge = self.k_edge(g[src_e]).view(-1, self.num_heads, self.head_dim)
        v_edge = self.v_edge(g[src_e]).view(-1, self.num_heads, self.head_dim)

        k_ctx = self.k_ctx(ctx).view(-1, self.num_heads, self.head_dim)
        v_ctx = self.v_ctx(ctx).view(-1, self.num_heads, self.head_dim)

        score_edge = (q * k_edge).sum(dim=-1) / math.sqrt(self.head_dim)
        score_ctx = (q * k_ctx).sum(dim=-1) / math.sqrt(self.head_dim)

        alpha = softmax(score_edge, dst_e, num_nodes=num_edges)
        beta = softmax(score_ctx, dst_e, num_nodes=num_edges)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        beta = F.dropout(beta, p=self.dropout, training=self.training)

        msg = alpha.unsqueeze(-1) * v_edge + beta.unsqueeze(-1) * v_ctx

        out = torch.zeros(
            num_edges,
            self.num_heads,
            self.head_dim,
            dtype=g.dtype,
            device=g.device,
        )
        out.index_add_(0, dst_e, msg)
        out = out.reshape(num_edges, self.out_dim)
        out = self.out_proj(out)

        z = self.norm1(self.res_proj(g) + out)
        z = self.norm2(z + self.ffn(z))
        return z


def _relation_type_source_to_target(
    source_src: int,
    source_dst: int,
    target_src: int,
    target_dst: int,
) -> tuple[int, int] | None:
    """
    Relation type between source edge f=(a,b) and target edge e=(u,v).

    Type IDs:
        0: head-to-head  b == v
        1: head-to-tail  b == u
        2: tail-to-head  a == v
        3: tail-to-tail  a == u

    Returns:
        (type_id, common_node)
    """
    if source_dst == target_dst:
        return 0, source_dst
    if source_dst == target_src:
        return 1, source_dst
    if source_src == target_dst:
        return 2, source_src
    if source_src == target_src:
        return 3, source_src
    return None


def _reverse_relation_type(t: Tensor) -> Tensor:
    """
    Reverse relation mapping:
        HH -> HH
        HT -> TH
        TH -> HT
        TT -> TT
    """
    mapping = torch.tensor([0, 2, 1, 3], dtype=torch.long, device=t.device)
    return mapping[t]


def build_augmented_dual_graph(
    edge_index: Tensor,
    edge_time: Optional[Tensor],
    mode: str = "augmented",
    causal_pruning: bool = True,
    max_dual_neighbors: int = 64,
) -> tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Builds GRANDE's augmented edge-adjacency graph inside a sampled subgraph.

    dual_edge_index[0] = source transaction edge
    dual_edge_index[1] = target transaction edge

    mode="augmented":
        connect two transaction edges if they share any endpoint.

    mode="line":
        ordinary directed line graph: source=(a,b), target=(b,c).

    causal_pruning:
        keep source_time < target_time.
    """
    device = edge_index.device
    src_list = edge_index[0].detach().cpu().tolist()
    dst_list = edge_index[1].detach().cpu().tolist()
    num_edges = len(src_list)

    if edge_time is not None:
        time_list = edge_time.detach().cpu().tolist()
    else:
        time_list = None

    incident: dict[int, list[int]] = defaultdict(list)
    outgoing: dict[int, list[int]] = defaultdict(list)

    for eid, (u, v) in enumerate(zip(src_list, dst_list)):
        incident[int(u)].append(eid)
        incident[int(v)].append(eid)
        outgoing[int(u)].append(eid)

    dual_src = []
    dual_dst = []
    dual_type = []
    dual_common = []
    dual_dt = []

    for target_eid, (u, v) in enumerate(zip(src_list, dst_list)):
        if mode == "line":
            candidate_edges = outgoing[int(u)]
            candidate_edges = [
                eid for eid in candidate_edges
                if dst_list[eid] == u or src_list[eid] == u
            ]
        else:
            candidate_edges = list(set(incident[int(u)] + incident[int(v)]))

        candidate_edges = [eid for eid in candidate_edges if eid != target_eid]

        if time_list is not None and len(candidate_edges) > max_dual_neighbors:
            candidate_edges = sorted(
                candidate_edges,
                key=lambda eid: abs(time_list[target_eid] - time_list[eid]),
            )[:max_dual_neighbors]
        elif len(candidate_edges) > max_dual_neighbors:
            candidate_edges = candidate_edges[:max_dual_neighbors]

        for source_eid in candidate_edges:
            if mode == "line":
                # Ordinary directed line graph: source=(a,b), target=(b,c)
                if dst_list[source_eid] != src_list[target_eid]:
                    continue

            if time_list is not None and causal_pruning:
                if time_list[source_eid] >= time_list[target_eid]:
                    continue

            rel = _relation_type_source_to_target(
                source_src=src_list[source_eid],
                source_dst=dst_list[source_eid],
                target_src=src_list[target_eid],
                target_dst=dst_list[target_eid],
            )
            if rel is None:
                continue

            type_id, common_node = rel

            dual_src.append(source_eid)
            dual_dst.append(target_eid)
            dual_type.append(type_id)
            dual_common.append(common_node)

            if time_list is not None:
                dual_dt.append(abs(time_list[target_eid] - time_list[source_eid]))

    if len(dual_src) == 0:
        dual_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        dual_edge_type = torch.empty((0,), dtype=torch.long, device=device)
        dual_common_node = torch.empty((0,), dtype=torch.long, device=device)
        dual_delta_t = None
        if edge_time is not None:
            dual_delta_t = torch.empty((0,), dtype=edge_time.dtype, device=device)
        return dual_edge_index, dual_edge_type, dual_common_node, dual_delta_t

    dual_edge_index = torch.tensor([dual_src, dual_dst], dtype=torch.long, device=device)
    dual_edge_type = torch.tensor(dual_type, dtype=torch.long, device=device)
    dual_common_node = torch.tensor(dual_common, dtype=torch.long, device=device)

    if edge_time is None:
        dual_delta_t = None
    else:
        dual_delta_t = torch.tensor(dual_dt, dtype=edge_time.dtype, device=device)

    return dual_edge_index, dual_edge_type, dual_common_node, dual_delta_t


def node_edge_time_features(
    edge_index_for_agg: Tensor,
    edge_time: Optional[Tensor],
    num_nodes: int,
    time_encoder: Optional[BochnerTimeEncoder],
) -> Optional[Tensor]:
    if edge_time is None or time_encoder is None:
        return None

    dst = edge_index_for_agg[1]
    inf = torch.full(
        (num_nodes,),
        float("inf"),
        dtype=edge_time.dtype,
        device=edge_time.device,
    )
    min_t = inf.scatter_reduce(
        0,
        dst,
        edge_time,
        reduce="amin",
        include_self=True,
    )
    dt = edge_time - min_t[dst]
    dt = dt.clamp_min(0.0)
    return time_encoder(dt)


class GRANDELayer(nn.Module):
    """
    One full GRANDE layer.

    Node update:
        h_in  = transformer over incoming neighborhood
        h_out = transformer over outgoing neighborhood
        h_next = merge(h_in, h_out)

    Edge update:
        g_in  = transformer over incoming dual neighbors
        g_out = transformer over outgoing dual neighbors
        g_next = merge(g_in, g_out)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        time_dim: int = 0,
        use_dual: bool = True,
    ):
        super().__init__()
        self.use_dual = use_dual

        self.node_in = GRANDEAttentionBlock(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            time_dim=time_dim,
        )
        self.node_out = GRANDEAttentionBlock(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            time_dim=time_dim,
        )

        self.node_merge = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_norm = nn.LayerNorm(hidden_dim)

        if use_dual:
            self.edge_in = GRANDEDualAttentionBlock(
                edge_dim=hidden_dim,
                node_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                time_dim=time_dim,
            )
            self.edge_out = GRANDEDualAttentionBlock(
                edge_dim=hidden_dim,
                node_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                time_dim=time_dim,
            )
            self.edge_merge = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.edge_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: Tensor,
        g: Tensor,
        edge_index: Tensor,
        edge_time_feat_in: Optional[Tensor],
        edge_time_feat_out: Optional[Tensor],
        dual_edge_index: Optional[Tensor],
        dual_edge_type: Optional[Tensor],
        dual_common_node: Optional[Tensor],
        dual_time_feat: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index
        num_nodes = h.size(0)
        num_edges = g.size(0)

        h_in = self.node_in(
            h=h,
            edge_index_for_agg=edge_index,
            g=g,
            dim_size=num_nodes,
            edge_time_feat=edge_time_feat_in,
        )

        reverse_edge_index = torch.stack([dst, src], dim=0)
        h_out = self.node_out(
            h=h,
            edge_index_for_agg=reverse_edge_index,
            g=g,
            dim_size=num_nodes,
            edge_time_feat=edge_time_feat_out,
        )

        h_next = self.node_norm(h + self.node_merge(torch.cat([h_in, h_out], dim=-1)))

        if not self.use_dual:
            return h_next, g

        g_in = self.edge_in(
            g=g,
            h=h_next,
            dual_edge_index=dual_edge_index,
            dual_edge_type=dual_edge_type,
            dual_common_node=dual_common_node,
            num_edges=num_edges,
            dual_time_feat=dual_time_feat,
        )

        reverse_dual_edge_index = dual_edge_index.flip(0)
        reverse_dual_edge_type = _reverse_relation_type(dual_edge_type)

        g_out = self.edge_out(
            g=g,
            h=h_next,
            dual_edge_index=reverse_dual_edge_index,
            dual_edge_type=reverse_dual_edge_type,
            dual_common_node=dual_common_node,
            num_edges=num_edges,
            dual_time_feat=dual_time_feat,
        )

        g_next = self.edge_norm(g + self.edge_merge(torch.cat([g_in, g_out], dim=-1)))
        return h_next, g_next


class CrossQueryAttention(nn.Module):
    """
    Exact GRANDE cross-query attention.

    For target edge (u, v):
        delta_uv = query h_u against v's incoming and outgoing neighborhoods
        delta_vu = query h_v against u's incoming and outgoing neighborhoods
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        max_cross_neighbors: int = 128,
    ):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for CrossQueryAttention.")

        self.hidden_dim = hidden_dim
        self.half = hidden_dim // 2
        self.max_cross_neighbors = max_cross_neighbors

        self.q = nn.Linear(hidden_dim, self.half)
        self.k_node = nn.Linear(hidden_dim, self.half)
        self.v_node = nn.Linear(hidden_dim, self.half)
        self.k_edge = nn.Linear(hidden_dim, self.half)
        self.v_edge = nn.Linear(hidden_dim, self.half)

        self.dropout = nn.Dropout(dropout)

    def _limit_neighbors(
        self,
        node_ids: Tensor,
        edge_repr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if node_ids.size(0) <= self.max_cross_neighbors:
            return node_ids, edge_repr
        node_ids = node_ids[-self.max_cross_neighbors :]
        edge_repr = edge_repr[-self.max_cross_neighbors :]
        return node_ids, edge_repr

    def _attend(
        self,
        query: Tensor,
        neighbor_h: Tensor,
        neighbor_g: Tensor,
    ) -> Tensor:
        if neighbor_h.size(0) == 0:
            return torch.zeros(self.half, dtype=query.dtype, device=query.device)

        q = self.q(query).view(1, -1)

        k_n = self.k_node(neighbor_h)
        v_n = self.v_node(neighbor_h)

        k_e = self.k_edge(neighbor_g)
        v_e = self.v_edge(neighbor_g)

        score_n = (q * k_n).sum(dim=-1) / math.sqrt(self.half)
        score_e = (q * k_e).sum(dim=-1) / math.sqrt(self.half)

        alpha = torch.softmax(score_n, dim=0)
        beta = torch.softmax(score_e, dim=0)

        alpha = self.dropout(alpha)
        beta = self.dropout(beta)

        return alpha @ v_n + beta @ v_e

    def forward(
        self,
        h: Tensor,
        g: Tensor,
        edge_index: Tensor,
        edge_label_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        src_all, dst_all = edge_index
        src_lab, dst_lab = edge_label_index

        delta_uv_list = []
        delta_vu_list = []

        for u, v in zip(src_lab.tolist(), dst_lab.tolist()):
            u_t = torch.tensor(u, dtype=torch.long, device=h.device)
            v_t = torch.tensor(v, dtype=torch.long, device=h.device)

            # Query u against v's incoming neighborhood: r -> v
            mask_v_in = dst_all == v_t
            neigh_v_in = src_all[mask_v_in]
            edge_v_in = g[mask_v_in]
            neigh_v_in, edge_v_in = self._limit_neighbors(neigh_v_in, edge_v_in)

            # Query u against v's outgoing neighborhood: v -> s
            mask_v_out = src_all == v_t
            neigh_v_out = dst_all[mask_v_out]
            edge_v_out = g[mask_v_out]
            neigh_v_out, edge_v_out = self._limit_neighbors(neigh_v_out, edge_v_out)

            q_u_to_v_in = self._attend(h[u], h[neigh_v_in], edge_v_in)
            q_u_to_v_out = self._attend(h[u], h[neigh_v_out], edge_v_out)
            delta_uv = torch.cat([q_u_to_v_in, q_u_to_v_out], dim=-1)

            # Query v against u's incoming neighborhood: r -> u
            mask_u_in = dst_all == u_t
            neigh_u_in = src_all[mask_u_in]
            edge_u_in = g[mask_u_in]
            neigh_u_in, edge_u_in = self._limit_neighbors(neigh_u_in, edge_u_in)

            # Query v against u's outgoing neighborhood: u -> s
            mask_u_out = src_all == u_t
            neigh_u_out = dst_all[mask_u_out]
            edge_u_out = g[mask_u_out]
            neigh_u_out, edge_u_out = self._limit_neighbors(neigh_u_out, edge_u_out)

            q_v_to_u_in = self._attend(h[v], h[neigh_u_in], edge_u_in)
            q_v_to_u_out = self._attend(h[v], h[neigh_u_out], edge_u_out)
            delta_vu = torch.cat([q_v_to_u_in, q_v_to_u_out], dim=-1)

            delta_uv_list.append(delta_uv)
            delta_vu_list.append(delta_vu)

        return torch.stack(delta_uv_list, dim=0), torch.stack(delta_vu_list, dim=0)


class GRANDEEdge(nn.Module):
    """
    Faithful GRANDE implementation for edge classification.

    Supported variants:
        grande
        grande_reduced
        grande_no_time
        grande_no_cross
        grande_no_pruning
        grande_line
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        time_dim: int = 32,
        variant: str = "grande",
        max_dual_neighbors: int = 64,
        max_cross_neighbors: int = 128,
    ):
        super().__init__()

        self.variant = variant.lower()
        self.hidden_dim = hidden_dim
        self.max_dual_neighbors = max_dual_neighbors

        self.use_dual = self.variant != "grande_reduced"
        self.use_cross = self.variant not in {"grande_reduced", "grande_no_cross"}
        self.use_time = self.variant != "grande_no_time"
        self.causal_pruning = self.variant != "grande_no_pruning"
        self.dual_mode = "line" if self.variant == "grande_line" else "augmented"

        effective_time_dim = time_dim if self.use_time else 0

        self.node_in = nn.Linear(num_node_features, hidden_dim)
        self.edge_in = nn.Linear(num_edge_features, hidden_dim)

        self.time_encoder = (
            BochnerTimeEncoder(effective_time_dim)
            if self.use_time and effective_time_dim > 0
            else None
        )

        self.layers = nn.ModuleList(
            [
                GRANDELayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    time_dim=effective_time_dim,
                    use_dual=self.use_dual,
                )
                for _ in range(num_layers)
            ]
        )

        if self.use_cross:
            self.cross_query = CrossQueryAttention(
                hidden_dim=hidden_dim,
                dropout=dropout,
                max_cross_neighbors=max_cross_neighbors,
            )
            head_in = hidden_dim * 5
        else:
            head_in = hidden_dim * 3

        self.edge_head = nn.Sequential(
            nn.Linear(head_in, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _dual_time_features(self, dual_delta_t: Optional[Tensor]) -> Optional[Tensor]:
        if not self.use_time or dual_delta_t is None or self.time_encoder is None:
            return None
        return self.time_encoder(dual_delta_t)

    def _target_edge_repr(
        self,
        g: Tensor,
        edge_id: Optional[Tensor],
        edge_label_id: Optional[Tensor],
        edge_label_attr: Tensor,
    ) -> Tensor:
        """
        Exact multigraph target-edge representation by edge ID.

        If a target edge is not present in the sampled message-passing graph,
        we fall back to the projected target edge features.
        """
        fallback = self.edge_in(edge_label_attr)

        if edge_id is None or edge_label_id is None:
            return fallback

        edge_id = edge_id.to(g.device).long()
        edge_label_id = edge_label_id.to(g.device).long()

        sorted_edge_id, perm = torch.sort(edge_id)
        pos = torch.searchsorted(sorted_edge_id, edge_label_id)

        valid = pos < sorted_edge_id.numel()
        safe_pos = pos.clamp(max=max(sorted_edge_id.numel() - 1, 0))

        if sorted_edge_id.numel() == 0:
            return fallback

        exact = sorted_edge_id[safe_pos] == edge_label_id
        valid = valid & exact

        out = fallback.clone()
        if valid.any():
            local_edge_pos = perm[safe_pos[valid]]
            out[valid] = g[local_edge_pos]

        return out

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_time: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        h = self.node_in(x)
        g = self.edge_in(edge_attr)

        num_nodes = h.size(0)

        if self.use_dual:
            dual_edge_index, dual_edge_type, dual_common_node, dual_delta_t = (
                build_augmented_dual_graph(
                    edge_index=edge_index,
                    edge_time=edge_time if self.use_time else None,
                    mode=self.dual_mode,
                    causal_pruning=self.causal_pruning,
                    max_dual_neighbors=self.max_dual_neighbors,
                )
            )
            dual_time_feat = self._dual_time_features(dual_delta_t)
        else:
            dual_edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
            dual_edge_type = torch.empty((0,), dtype=torch.long, device=x.device)
            dual_common_node = torch.empty((0,), dtype=torch.long, device=x.device)
            dual_time_feat = None

        src, dst = edge_index
        reverse_edge_index = torch.stack([dst, src], dim=0)

        edge_time_feat_in = node_edge_time_features(
            edge_index_for_agg=edge_index,
            edge_time=edge_time if self.use_time else None,
            num_nodes=num_nodes,
            time_encoder=self.time_encoder,
        )

        edge_time_feat_out = node_edge_time_features(
            edge_index_for_agg=reverse_edge_index,
            edge_time=edge_time if self.use_time else None,
            num_nodes=num_nodes,
            time_encoder=self.time_encoder,
        )

        for layer in self.layers:
            h, g = layer(
                h=h,
                g=g,
                edge_index=edge_index,
                edge_time_feat_in=edge_time_feat_in,
                edge_time_feat_out=edge_time_feat_out,
                dual_edge_index=dual_edge_index,
                dual_edge_type=dual_edge_type,
                dual_common_node=dual_common_node,
                dual_time_feat=dual_time_feat,
            )

        return h, g

    def decode(
        self,
        h: Tensor,
        g: Tensor,
        edge_index: Tensor,
        edge_label_index: Tensor,
        edge_label_attr: Tensor,
        edge_id: Optional[Tensor] = None,
        edge_label_id: Optional[Tensor] = None,
    ) -> Tensor:
        src_lab, dst_lab = edge_label_index

        h_src = h[src_lab]
        h_dst = h[dst_lab]

        g_lab = self._target_edge_repr(
            g=g,
            edge_id=edge_id,
            edge_label_id=edge_label_id,
            edge_label_attr=edge_label_attr,
        )

        if self.use_cross:
            delta_uv, delta_vu = self.cross_query(
                h=h,
                g=g,
                edge_index=edge_index,
                edge_label_index=edge_label_index,
            )
            z = torch.cat([g_lab, h_dst, h_src, delta_uv, delta_vu], dim=-1)
        else:
            z = torch.cat([g_lab, h_dst, h_src], dim=-1)

        return self.edge_head(z)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_label_index: Tensor,
        edge_label_attr: Tensor,
        edge_id: Optional[Tensor] = None,
        edge_label_id: Optional[Tensor] = None,
        edge_time: Optional[Tensor] = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        h, g = self.encode(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_time=edge_time,
        )

        logits = self.decode(
            h=h,
            g=g,
            edge_index=edge_index,
            edge_label_index=edge_label_index,
            edge_label_attr=edge_label_attr,
            edge_id=edge_id,
            edge_label_id=edge_label_id,
        )

        return logits, (h, g)