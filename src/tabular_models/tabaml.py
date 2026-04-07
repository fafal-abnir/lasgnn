from __future__ import annotations

import math

import torch
from torch import nn


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 0.01):
    with torch.no_grad():
        return tensor.normal_(mean, std).clamp_(mean - 2 * std, mean + 2 * std)


class SharedCategoryEmbedding(nn.Module):
    """
    Column embedding + shared embedding component.
    Shared-to-individual ratio defaults to 1/8 as reported best in the paper.
    """
    def __init__(self, cardinality: int, dim: int, shared_embed_ratio: float = 0.125):
        super().__init__()
        shared_dim = max(1, int(round(dim * shared_embed_ratio)))
        indiv_dim = dim - shared_dim
        self.shared = nn.Parameter(torch.zeros(1, shared_dim))
        self.indiv = nn.Embedding(cardinality, indiv_dim)

        trunc_normal_(self.shared, std=0.01)
        trunc_normal_(self.indiv.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indiv = self.indiv(x)
        shared = self.shared.expand(x.size(0), -1)
        return torch.cat([indiv, shared], dim=-1)


class ResidualMultiheadSelfAttention(nn.Module):
    """
    Residual attention:
        softmax(QK^T / sqrt(dk) + prev_scores) V
    """
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        prev_scores: torch.Tensor | None = None,
    ):
        b, n, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, Hd]
        k = k.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,N,N]

        if prev_scores is not None:
            scores = scores + prev_scores

        if attn_mask is not None:
            # attn_mask: [N, N], bool, True means allow
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), mask_value)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B,H,N,Hd]
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        out = self.out_proj(out)
        return out, scores


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ResidualAttentionEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ResidualMultiheadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        prev_scores: torch.Tensor | None = None,
    ):
        h, scores = self.attn(self.norm1(x), attn_mask=attn_mask, prev_scores=prev_scores)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x, scores


class EncoderStack(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualAttentionEncoderLayer(dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        prev_scores = None
        for layer in self.layers:
            x, prev_scores = layer(x, attn_mask=attn_mask, prev_scores=prev_scores)
        return x


class TabAML(nn.Module):
    """
    More faithful Tab-AML-style model:
    - column-based categorical embeddings
    - shared embedding component
    - dual-masked transformer encoders
    - residual attention
    - MLP over flattened contextual categorical embeddings + continuous features
    """

    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: list[int],
        cat_feature_names: list[str],
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        shared_embed_ratio: float = 0.125,
        mlp_hidden_mult: int = 4,
    ):
        super().__init__()

        self.num_numeric = num_numeric
        self.cat_cardinalities = cat_cardinalities
        self.cat_feature_names = cat_feature_names
        self.hidden_dim = hidden_dim

        self.cat_embs = nn.ModuleList(
            [
                SharedCategoryEmbedding(cardinality, hidden_dim, shared_embed_ratio=shared_embed_ratio)
                for cardinality in cat_cardinalities
            ]
        )

        self.encoder_masked = EncoderStack(hidden_dim, num_heads, num_layers=max(1, num_layers // 2), dropout=dropout)
        self.encoder_unmasked = EncoderStack(hidden_dim, num_heads, num_layers=max(1, num_layers - max(1, num_layers // 2)), dropout=dropout)

        # build mask for first encoder: only sender_account and receiver_account can attend
        num_cat = len(cat_feature_names)
        mask = torch.zeros((num_cat, num_cat), dtype=torch.bool)
        if "sender_account" in cat_feature_names and "receiver_account" in cat_feature_names:
            i = cat_feature_names.index("sender_account")
            j = cat_feature_names.index("receiver_account")
            mask[i, i] = True
            mask[j, j] = True
            mask[i, j] = True
            mask[j, i] = True
        else:
            # fallback: identity if sender/receiver unavailable
            mask.fill_diagonal_(True)
        self.register_buffer("first_stage_mask", mask)

        input_dim = hidden_dim * num_cat + num_numeric
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * mlp_hidden_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * mlp_hidden_mult, max(input_dim // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(input_dim // 2, 1), 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        if x_cat.size(1) != len(self.cat_embs):
            raise ValueError("Mismatch between x_cat columns and cat embeddings")

        cat_tokens = []
        for i, emb in enumerate(self.cat_embs):
            cat_tokens.append(emb(x_cat[:, i]))
        x = torch.stack(cat_tokens, dim=1)  # [B, num_cat, d]

        x = self.encoder_masked(x, attn_mask=self.first_stage_mask)
        x = self.encoder_unmasked(x, attn_mask=None)

        x = x.flatten(start_dim=1)
        if x_num.numel() > 0:
            x = torch.cat([x, x_num], dim=-1)

        return self.mlp(x)