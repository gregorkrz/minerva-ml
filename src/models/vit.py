import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Positional encodings (absolute, from coordinates) --------------

class FourierPositionalEncoding(nn.Module):
    def __init__(self, coord_dim: int, num_bands: int = 16, max_freq: float = 10.0):
        super().__init__()
        self.register_buffer("freq_bands", torch.linspace(1.0, max_freq, num_bands), persistent=False)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        # pos: [B, N, D] -> [B, N, D*num_bands*2]
        x = pos.unsqueeze(-1) * self.freq_bands.view(1, 1, 1, -1)  # [B,N,D,K]
        pe = torch.cat([torch.sin(2 * math.pi * x), torch.cos(2 * math.pi * x)], dim=-1)
        return pe.flatten(start_dim=-2)


class AbsolutePositionalMLP(nn.Module):
    def __init__(self, coord_dim: int, d_model: int, num_bands: int = 16, max_freq: float = 10.0):
        super().__init__()
        self.fourier = FourierPositionalEncoding(coord_dim, num_bands=num_bands, max_freq=max_freq)
        in_dim = coord_dim * num_bands * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.fourier(pos))


# ---------------- FlashAttention-backed MHSA (via PyTorch SDPA) ------------------

class FlashMHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # [B,h,N,dh]
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.proj(out)
        return self.proj_drop(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = FlashMHSA(d_model=d_model, n_heads=n_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------- Mixed continuous + categorical feature encoder -----------------

class MixedFeatureEncoder(nn.Module):
    """
    Encodes continuous + multiple categorical fields into a single d_model vector.

    continuous: [B, ..., C_cont]  (can be [B,N,C] or [B,C])
    categorical: list of tensors, each [B, ...] with dtype long (class indices)

    Each categorical field i has num_classes[i] and emb_dim[i].
    """
    def __init__(
        self,
        cont_dim: int,
        cat_num_classes: List[int],
        d_model: int,
        cat_emb_dim: int = 16,
        cont_hidden: Optional[int] = None,
        dropout: float = 0.0,
        use_cont_layernorm: bool = True,
    ):
        super().__init__()
        self.cont_dim = cont_dim
        self.cat_num_classes = cat_num_classes
        self.d_model = d_model

        # Continuous pathway
        self.cont_norm = nn.LayerNorm(cont_dim) if (use_cont_layernorm and cont_dim > 0) else None
        if cont_dim > 0:
            h = cont_hidden if cont_hidden is not None else max(d_model, cont_dim)
            self.cont_mlp = nn.Sequential(
                nn.Linear(cont_dim, h),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(h, d_model),
            )
        else:
            self.cont_mlp = None

        # Categorical pathway: one embedding per field
        self.cat_embs = nn.ModuleList([
            nn.Embedding(nc, cat_emb_dim) for nc in cat_num_classes
        ]) if len(cat_num_classes) > 0 else nn.ModuleList()

        cat_total = cat_emb_dim * len(cat_num_classes)
        self.cat_proj = nn.Linear(cat_total, d_model) if cat_total > 0 else None

        # Final combine
        # We combine by summing cont and cat representations (both are d_model).
        # (If only one exists, it's used directly.)
        self.out_drop = nn.Dropout(dropout)

    def forward(
        self,
        cont: Optional[torch.Tensor],
        cats: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Returns: [B, ..., d_model]
        """
        reps = []

        if self.cont_mlp is not None:
            x = cont
            if self.cont_norm is not None:
                x = self.cont_norm(x)
            reps.append(self.cont_mlp(x))

        if self.cat_proj is not None:
            if cats is None:
                raise ValueError("cats must be provided because categorical fields were configured.")
            if len(cats) != len(self.cat_embs):
                raise ValueError(f"Expected {len(self.cat_embs)} categorical tensors, got {len(cats)}.")

            emb_list = []
            for t, emb in zip(cats, self.cat_embs):
                if t.dtype != torch.long:
                    t = t.long()
                emb_list.append(emb(t))  # [B,...,cat_emb_dim]
            cat = torch.cat(emb_list, dim=-1)  # [B,...,cat_total]
            reps.append(self.cat_proj(cat))    # [B,...,d_model]

        if len(reps) == 0:
            raise ValueError("No continuous or categorical features configured.")
        out = reps[0]
        for r in reps[1:]:
            out = out + r
        return self.out_drop(out)


# ---------------- Point + Global ViT with mixed feature types -------------------

@dataclass
class PointGlobalMixedViTConfig:
    # point inputs
    point_cont_dim: int
    point_cat_num_classes: List[int]         # one entry per categorical field
    # global/event inputs
    global_cont_dim: int
    global_cat_num_classes: List[int]

    coord_dim: int = 3
    d_model: int = 256
    depth: int = 6
    n_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    use_cls_token: bool = True
    use_event_token: bool = True

    # embedding knobs
    cat_emb_dim: int = 16
    cont_hidden: Optional[int] = None

    # position encoding knobs
    num_bands: int = 16
    max_freq: float = 10.0


class PointGlobalMixedViT(nn.Module):
    """
    Inputs:
      point_cont: [B, N, point_cont_dim] or None if dim=0
      point_cats: list of [B, N] long tensors (len = len(point_cat_num_classes)) or None
      pos:        [B, N, coord_dim]

      global_cont: [B, global_cont_dim] or None if dim=0
      global_cats: list of [B] long tensors (len = len(global_cat_num_classes)) or None

    Output:
      if CLS: [B, d_model] else [B, T, d_model]
    """
    def __init__(self, cfg: PointGlobalMixedViTConfig):
        super().__init__()
        self.cfg = cfg

        self.point_encoder = MixedFeatureEncoder(
            cont_dim=cfg.point_cont_dim,
            cat_num_classes=cfg.point_cat_num_classes,
            d_model=cfg.d_model,
            cat_emb_dim=cfg.cat_emb_dim,
            cont_hidden=cfg.cont_hidden,
            dropout=cfg.dropout,
        )

        self.pos_enc = AbsolutePositionalMLP(
            coord_dim=cfg.coord_dim,
            d_model=cfg.d_model,
            num_bands=cfg.num_bands,
            max_freq=cfg.max_freq,
        )

        # CLS token
        self.cls = None
        if cfg.use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)

        # EVT token from global mixed features
        self.global_encoder = None
        self.evt_bias = None
        if cfg.use_event_token:
            self.global_encoder = MixedFeatureEncoder(
                cont_dim=cfg.global_cont_dim,
                cat_num_classes=cfg.global_cat_num_classes,
                d_model=cfg.d_model,
                cat_emb_dim=cfg.cat_emb_dim,
                cont_hidden=cfg.cont_hidden,
                dropout=cfg.dropout,
            )
            self.evt_bias = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
            nn.init.trunc_normal_(self.evt_bias, std=0.02)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            EncoderBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        point_cont: Optional[torch.Tensor],
        point_cats: Optional[List[torch.Tensor]],
        pos: torch.Tensor,
        global_cont: Optional[torch.Tensor],
        global_cats: Optional[List[torch.Tensor]],
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = pos.shape

        # Point tokens: [B,N,d]
        x = self.point_encoder(point_cont, point_cats) + self.pos_enc(pos)

        # Special tokens
        toks = []
        if self.cls is not None:
            toks.append(self.cls.expand(B, 1, -1))

        if self.global_encoder is not None:
            g = self.global_encoder(global_cont, global_cats)  # [B,d]
            toks.append(g.unsqueeze(1) + self.evt_bias)        # [B,1,d]

        if toks:
            x = torch.cat([*toks, x], dim=1)  # [B, N + n_special, d]

        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.norm(x)

        return x[:, 0] if self.cls is not None else x


# ---------------- Example ----------------
if __name__ == "__main__":
    # Example:
    # per-point continuous: 6 floats
    # per-point categorical: [particle_type (10 classes), layer_id (50 classes)]
    # Global continuous: 4 floats
    # Global categorical: [run_period (5 classes)]
    cfg = PointGlobalMixedViTConfig(
        point_cont_dim=4,
        point_cat_num_classes=[8],
        global_cont_dim=4,
        global_cat_num_classes=[],
        coord_dim=2,
        d_model=128,
        depth=4,
        n_heads=4,
        use_cls_token=True,
        use_event_token=True,
        cat_emb_dim=16,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PointGlobalMixedViT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params}")

    B, N = 8, 33
    pos = torch.randn(B, N, cfg.coord_dim, device=device)

    point_cont = torch.randn(B, N, cfg.point_cont_dim, device=device)
    point_cats = [
        torch.randint(0, 8, (B, N), device=device),   # particle_type
    ]

    global_cont = torch.randn(B, cfg.global_cont_dim, device=device)
    global_cats =  None
    out = model(point_cont, point_cats, pos, global_cont, global_cats)

    print(out.shape)  # [B, d_model]
    print(out)

