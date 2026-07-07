"""ParticleViT_Pool: attention pooling instead of a CLS token."""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.hyperscale.common import (
    TransformerBlock,
    _zero_masked_tokens,
    init_olmo_weights,
)


class AttentionPool(nn.Module):
    """Single learned query attends over the token sequence. Replaces a CLS token."""

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert embedding_dim % n_heads == 0, "Embedding dim is not divisible by nheads"

        factory = {"device": device, "dtype": dtype}
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        self.query = nn.Parameter(torch.empty(1, 1, embedding_dim))
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False, **factory)
        self.kv_proj = nn.Linear(
            embedding_dim, 2 * embedding_dim, bias=False, **factory
        )
        self.q_norm = nn.RMSNorm(self.head_dim, **factory)
        self.k_norm = nn.RMSNorm(self.head_dim, **factory)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False, **factory)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        q_in = self.query.expand(B, -1, -1)

        q = einops.rearrange(self.q_proj(q_in), "b l (h d) -> b h l d", h=self.n_heads)
        kv = einops.rearrange(
            self.kv_proj(x),
            "b l (kv h d) -> kv b h l d",
            kv=2,
            h=self.n_heads,
        )
        k, v = kv.unbind(0)

        q = self.q_norm(q)
        k = self.k_norm(k)

        key_mask = None
        if attn_mask is not None:
            key_mask = einops.rearrange(attn_mask.bool(), "b l -> b 1 1 l")

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        out = einops.rearrange(out, "b h l d -> b l (h d)")
        out = self.out_proj(out).squeeze(1)
        return out


class ParticleVIT_Pool(nn.Module):
    """ParticleViT variant with attention pooling instead of a CLS token."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.token_embed = nn.Linear(num_features, embed_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    num_features=num_features,
                    num_classes=num_classes,
                    embedding_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.RMSNorm(embed_dim)
        self.pool = AttentionPool(embed_dim, num_heads)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(init_olmo_weights)
        nn.init.trunc_normal_(self.pool.query, mean=0.0, std=0.02, a=-0.06, b=0.06)

    def forward(
        self,
        X: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embed(X)
        x = _zero_masked_tokens(x, attn_mask)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.norm(x)
        x = _zero_masked_tokens(x, attn_mask)
        pooled = self.pool(x, attn_mask=attn_mask)
        logits = self.head(pooled)
        return logits
