# Models used in this repository

This document summarizes the model families used by `src/scripts/train.py`.

## Overview

The training script supports:

- **PointGlobalMixedViT** (`Transformer1` style runs)
- **OmniLearned PET2** (`--use-omnilearned small|medium|large`)
- **CondOnlyMLP** baseline (`--cond_only`)

Tasks are either:

- **Regression** (e.g., `-E-available-no-muon`)
- **Classification** (e.g., `-npi2`)

## 1) PointGlobalMixedViT

Implementation: `src/models/vit.py` (`PointGlobalMixedViT`).

### Inputs

Per event:
- point features (continuous + optional PID embedding)
- 2D coordinates for positional encoding
- optional global/event-level features

A CLS token is used for readout, and an event token can be added for global conditioning.

### Attention

The model uses multi-head self-attention via PyTorch scaled dot-product attention.

For one head:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
$$

Multi-head output is concatenated and projected:

$$
\mathrm{MHSA}(X)=\mathrm{Concat}(\text{head}_1,\dots,\text{head}_H)W_O
$$

### Transformer block

Each encoder block is pre-norm residual attention + MLP:

$$
X' = X + \mathrm{MHSA}(\mathrm{LN}(X)), \quad
Y = X' + \mathrm{MLP}(\mathrm{LN}(X'))
$$

with MLP hidden size approximately `mlp_ratio * d_model`.

### Typical settings in job submission

`src/jobs/submit_train_jobs.py` uses (for `Transformer1`):

- `d_model = 128`
- `depth = 4`
- `n_heads = 8`
- `dropout = 0.0`
- `attn_dropout = 0.0`

## 2) OmniLearned PET2 variants

Implementation: `src/models/omnilearned/network.py`, created in `create_omnilearned_model`.

Enabled via:

```bash
--use-omnilearned small
--use-omnilearned medium
--use-omnilearned large
```

Parameter presets are defined in `src/models/omnilearned/utils.py`:

- **small**: base_dim 128, num_heads 8, num_transformers 8
- **medium**: base_dim 512, num_heads 16, num_transformers 12
- **large**: base_dim 1024, num_heads 32, num_transformers 28

In `src/jobs/submit_train_jobs.py`, the `model` identifiers in the `for model in [...]` loop map to CLI flags as:

- `OLS` -> OmniLearned small + pretrained (`--use-pretrained pretrain_s`)
- `OLS_RW` -> OmniLearned small from random initialization (`--use-omnilearned small` with no `--use-pretrained`)
- `OLM` -> OmniLearned medium + pretrained (`--use-pretrained pretrain_m`)

## 3) CondOnlyMLP baseline

Implementation: `CondOnlyMLP` in `src/scripts/train.py`.

Used with `--cond_only`, this model ignores particle tokens and uses only global conditioning features. It is an MLP with residual blocks:

$$
h_0 = \mathrm{InputProj}(c), \quad h_{\ell+1}=h_\ell + f_\ell(\mathrm{LN}(h_\ell))
$$

and final prediction head:

$$
\hat{y}=W_2\,\sigma(W_1\,\mathrm{LN}(h_L))
$$

For regression, the output can be constrained positive using `softplus`.

## Output heads and tasks

The final head is task-dependent:

- Regression: linear head to one scalar
- Classification: linear head to `N_classes` logits

Conceptually:

$$
\hat{y} = W_{\text{head}} z + b
$$

where $z$ is the final CLS/event representation.
