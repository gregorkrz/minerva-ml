"""HyperScale ParticleViT models, vendored from gregorkrz/HyperScale.

Three variants over particle-set inputs (no positional encoding; reordered-norm + QK-Norm
+ SwiGLU transformer blocks):
    - ParticleVIT: simple linear input embedding + CLS token.
    - ParticleVIT_Embedding: split kinematics/PID/vertex embeddings (requires 9 input features).
    - ParticleVIT_Pool: linear input embedding + learned attention pool (no CLS token).
"""

from src.models.hyperscale.common import (
    MultiHeadAttention,
    PackedSwiGLU,
    TransformerBlock,
    _zero_masked_tokens,
    init_olmo_weights,
)
from src.models.hyperscale.particle_vit import ParticleVIT
from src.models.hyperscale.particle_vit_embedding import (
    ParticleInputEmbedding,
    ParticleVIT_Embedding,
)
from src.models.hyperscale.particle_vit_pool import AttentionPool, ParticleVIT_Pool
from src.models.hyperscale.utils import (
    load_pretrained_hyperscale,
    peek_hyperscale_checkpoint_args,
)

__all__ = [
    "ParticleVIT",
    "ParticleVIT_Embedding",
    "ParticleVIT_Pool",
    "ParticleInputEmbedding",
    "AttentionPool",
    "TransformerBlock",
    "MultiHeadAttention",
    "PackedSwiGLU",
    "init_olmo_weights",
    "_zero_masked_tokens",
    "load_pretrained_hyperscale",
    "peek_hyperscale_checkpoint_args",
]
