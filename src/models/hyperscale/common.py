"""Shared HyperScale building blocks (TransformerBlock, MultiHeadAttention, SwiGLU, init).

Vendored verbatim from https://github.com/gregorkrz/HyperScale (modeling/models/common.py).
"""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_masked_tokens(
    X: torch.Tensor,
    attn_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Zero out tokens where attn_mask is False (i.e. padding positions)."""
    if attn_mask is None:
        return X
    return X.masked_fill(~attn_mask.unsqueeze(-1), 0.0)


def init_olmo_weights(module: nn.Module) -> None:
    """OLMo 2/3 init: trunc_normal(0, 0.02) clipped at +/-3 std on linears,
    zero on biases, ones on norm gammas. No depth-scaled residual init --
    stability comes from the reordered-norm + QK-Norm recipe instead.
    Ref: OLMo 2 paper (arxiv 2501.00656, sec 3); OLMo-core init.py.
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.06, b=0.06)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.RMSNorm):
        nn.init.ones_(module.weight)


class PackedSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 8 / 3,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.w13 = nn.Linear(dim, 2 * hidden_dim, bias=False, **factory_kwargs)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)

    def forward(self, x):
        x1, x3 = torch.chunk(self.w13(x), 2, dim=-1)
        return self.w2(F.silu(x1) * x3)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with QK-Norm using PyTorch SDPA."""

    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        assert embedding_dim % n_heads == 0, "Embedding dim is not divisible by nheads"

        self.n_heads = n_heads
        self.embedding_dim_head = embedding_dim // n_heads

        self.packed_input_projection = nn.Linear(
            embedding_dim,
            embedding_dim * 3,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.q_norm = nn.RMSNorm(
            self.embedding_dim_head,
            device=device,
            dtype=dtype,
        )
        self.k_norm = nn.RMSNorm(
            self.embedding_dim_head,
            device=device,
            dtype=dtype,
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        X: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv: torch.Tensor = einops.rearrange(
            self.packed_input_projection(X),
            "bs l (qkv h e) -> qkv bs h l e",
            qkv=3,
            h=self.n_heads,
        )
        Q, K, V = qkv.unbind(0)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        key_attention_mask = None
        query_pad_mask = None
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=torch.bool)
            key_attention_mask = einops.rearrange(attn_mask, "bs l -> bs 1 1 l")
            query_pad_mask = einops.rearrange(~attn_mask, "bs l -> bs 1 l 1")

        attn_output = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            attn_mask=key_attention_mask,
        )

        if query_pad_mask is not None:
            attn_output = attn_output.masked_fill(query_pad_mask, 0.0)

        attn_output = einops.rearrange(attn_output, "bs h l e -> bs l (h e)")
        attn_output = self.output_projection(attn_output)

        return attn_output


class TransformerBlock(nn.Module):
    """HyperScale Transformer block: reordered-norm (double norm) + QK-Norm + SwiGLU."""

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        embedding_dim: int,
        num_heads: int,
        mlp_ratio: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.embed_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.device = device
        self.dtype = dtype

        self.multihead_attn = MultiHeadAttention(
            embedding_dim=embedding_dim,
            n_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        self.ffn = PackedSwiGLU(
            dim=embedding_dim,
            mlp_ratio=mlp_ratio,
            device=device,
            dtype=dtype,
        )
        self.pre_attn_norm = nn.RMSNorm(embedding_dim, device=device, dtype=dtype)
        self.post_attn_norm = nn.RMSNorm(embedding_dim, device=device, dtype=dtype)
        self.pre_ffn_norm = nn.RMSNorm(embedding_dim, device=device, dtype=dtype)
        self.post_ffn_norm = nn.RMSNorm(embedding_dim, device=device, dtype=dtype)

    def forward(
        self,
        X: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        X_block = self.pre_attn_norm(X)
        X_block = self.multihead_attn(X_block, attn_mask=attn_mask)
        X_block = self.post_attn_norm(X_block)
        X = X + X_block

        X_ffn = self.pre_ffn_norm(X)
        X_ffn = self.ffn(X_ffn)
        X_ffn = self.post_ffn_norm(X_ffn)
        X = X + X_ffn

        X = _zero_masked_tokens(X, attn_mask)
        return X
