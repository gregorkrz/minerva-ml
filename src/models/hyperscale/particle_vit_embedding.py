"""ParticleViT with split input embeddings for the 9 OmniLearned features
(4 kinematics + 1 PID + 4 vertex)."""

import torch
import torch.nn as nn

from src.models.hyperscale.common import (
    TransformerBlock,
    _zero_masked_tokens,
    init_olmo_weights,
)


class ParticleInputEmbedding(nn.Module):
    """Embed kinematics, PID, and vertex features through separate paths."""

    def __init__(
        self,
        num_features: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if num_features != 9:
            raise ValueError(
                f"ParticleInputEmbedding expects 9 features, got {num_features}",
            )

        self.kin_embed = nn.Linear(4, embed_dim)
        self.pid_embed = nn.Embedding(9, embed_dim)
        self.vertex_embed = nn.Linear(4, embed_dim)
        self.pid_available_embed = nn.Embedding(2, embed_dim)
        self.vertex_available_embed = nn.Embedding(2, embed_dim)

    def reset_parameters(self) -> None:
        init_olmo_weights(self.kin_embed)
        init_olmo_weights(self.vertex_embed)
        nn.init.trunc_normal_(
            self.pid_embed.weight,
            mean=0.0,
            std=0.02,
            a=-0.06,
            b=0.06,
        )
        nn.init.zeros_(self.pid_available_embed.weight)
        nn.init.zeros_(self.vertex_available_embed.weight)

    def forward(
        self,
        X: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attn_mask is None:
            real_mask = X[:, :, 2] != 0
        else:
            real_mask = attn_mask.bool()

        vertex_available = (X[:, :, 5:9] != 0).any(dim=-1) & real_mask
        pid_available = vertex_available.any(dim=1, keepdim=True).expand_as(real_mask)

        x = self.kin_embed(X[:, :, :4])
        dtype = x.dtype
        x = x + self.pid_embed(X[:, :, 4].long()).to(dtype=dtype)
        x = x + self.vertex_embed(X[:, :, 5:9])
        x = x + self.pid_available_embed(pid_available.long()).to(dtype=dtype)
        x = x + self.vertex_available_embed(vertex_available.long()).to(dtype=dtype)
        return x


class ParticleVIT_Embedding(nn.Module):
    """ParticleViT with split input embeddings for the 9 OmniLearned features."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.token_embed = ParticleInputEmbedding(
            num_features=num_features,
            embed_dim=embed_dim,
        )
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
            ],
        )
        self.norm = nn.RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(init_olmo_weights)
        self.token_embed.reset_parameters()
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-0.06, b=0.06)

    def forward(
        self,
        X: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.token_embed(X, attn_mask=attn_mask)
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
