"""ParticleViT (basic): single CLS token, simple linear input embedding."""

import torch
import torch.nn as nn

from src.models.hyperscale.common import (
    TransformerBlock,
    _zero_masked_tokens,
    init_olmo_weights,
)


class ParticleVIT(nn.Module):
    """ParticleViT variant with a single class token instead of attention pooling."""

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
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))

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
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(init_olmo_weights)
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-0.06, b=0.06)

    def forward(
        self,
        X: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embed(X)
        x = _zero_masked_tokens(x, attn_mask)
        bs = x.shape[0]

        cls_token = self.cls_token.expand(bs, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        if attn_mask is not None:
            cls_mask = torch.ones((bs, 1), dtype=torch.bool, device=attn_mask.device)
            attn_mask = torch.cat([cls_mask, attn_mask], dim=1)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.norm(x)
        x = _zero_masked_tokens(x, attn_mask)
        cls = x[:, 0]
        logits = self.head(cls)
        return logits
