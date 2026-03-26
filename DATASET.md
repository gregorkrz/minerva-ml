# MINERvA ML-Ready Dataset

This document describes the ML-ready dataset produced by the preprocessing pipeline
(`src/scripts/preprocess_dataset.py` → `src/scripts/split_dataset.py`).

The raw data comes from the **MINERvA Medium Energy Forward Horn Current (FHC) Standard Monte Carlo**
simulation, organized into playlists (1A, 1B, ..., 1P). Each playlist corresponds to a
distinct data-taking period.

---

## File Format

Each split is stored as a PyTorch file (`.pb`) containing a dictionary with three keys:

| Key | Type | Shape |
|---|---|---|
| `data` | `torch.nested.nested_tensor` (jagged) | `(N_events, variable N_particles, 10)` |
| `truth_labels` | `numpy.ndarray` | `(N_events, 15)` |
| `global_features` | `numpy.ndarray` | `(N_events, 13)` |

The dataset is split into `train/`, `val/`, and `test/` directories (default 80/10/10 split, stratified by interaction type). Only events with interaction types in {1, 2, 3, 4, 8} are kept; all others are filtered out.

---

## Per-Particle Features (`data`)

Each event is represented as a **variable-length point cloud** of reconstructed particles.
Every particle carries a 10-dimensional feature vector, split into two groups:

### Columns 0–4: Kinematic features

| Index | Name | Description |
|---|---|---|
| 0 | η | Pseudorapidity, computed from the 3-momentum (clipped to [−10, 10]) |
| 1 | φ | Azimuthal angle (radians), from `atan2(py, px)` |
| 2 | log(p_T) | Log of transverse momentum: `log(pT + 1e-6)` |
| 3 | log(E) | Log of energy: `log(E + 1e-6)` |
| 4 | PID | Integer node type (see table below) |

The kinematic features are derived from a four-momentum `(px, py, pz, E)` via `convert_four_momentum`. The meaning of `(px, py, pz, E)` depends on the node type—see the node-type descriptions below.

### Columns 5–9: Additional features

| Index | Name | Description |
|---|---|---|
| 5 | dE/dx | Preprocessed mean energy loss (see below); 0 for muons and blobs |
| 6 | x | Spatial x-coordinate / 10000; 0 for muons and photons |
| 7 | y | Spatial y-coordinate / 10000; 0 for muons and photons |
| 8 | z | Spatial z-coordinate / 10000; 0 for muons and photons |
| 9 | t | Time / 10000; available for all node types |

**dE/dx preprocessing:** Raw dE/dx values are cleaned (−999 → 0, values > 100 → 100), then transformed as `log(|dE/dx| + 0.1)`.

**Coordinate preprocessing:** Raw spatial coordinates and times are divided by 10000.

---

## Node Types (PID)

Each particle in the point cloud is assigned an integer PID that encodes its reconstruction category.

| PID | Type | Max per event | Description |
|---|---|---|---|
| 0 | **Muon** | 1 | MINOS-matched reconstructed muon |
| 1 | **Photon** | 2 | Reconstructed photon (from `gamma1` / `gamma2` branches) |
| 2 | **Blob** | 20 | Calorimetric energy deposit (no particle hypothesis) |
| 3 | **Prong (pion)** | — | Reconstructed prong with pion hypothesis (original `prong_part_pid = 3`) |
| 4 | **Prong (EM shower)** | — | Reconstructed prong with EM shower hypothesis (original `prong_part_pid = 8`) |
| 5 | **Prong (muon-like)** | — | Reconstructed prong with muon hypothesis (original `prong_part_pid = 13`) |
| 6 | **Aggregated blob** | 1 | Sum of all blobs beyond the top-19 by energy; four-momenta are summed, additional features are averaged |
| 7 | **Aggregated prong** | 1 | Sum of all prongs beyond the top-9 by energy; same aggregation as blobs |

With the default settings (`--max-blobs 20 --max-prongs 10`), the maximum number of particles per event is **33** (1 muon + 2 photons + 20 blobs + 10 prongs).

### Muon (PID 0)

Source: `muon_corrected_p` (MINOS-matched corrected four-momentum).

Only events with a valid MINOS-matched muon are included (both `muon_corrected_p[:, 0]` and `muon_corrected_p[:, 1]` must be ≠ −999). Muons with any component |value| > 10⁶ are removed as anomalies.

- **Four-momentum** `(px, py, pz, E)`: taken directly from `muon_corrected_p`.
- **Additional features**: `[0, 0, 0, 0, time]` where time is `muon_trackVertexTime / 10000`.

### Photon (PID 1)

Source: `gamma1_*` and `gamma2_*` branches (the two leading reconstructed photon candidates from π⁰ → γγ reconstruction).

Only photons with `E > 1e-5` are kept. Each event has 0, 1, or 2 photons.

- **Four-momentum** `(px, py, pz, E)`: from `gamma{1,2}_{px, py, pz, E}`.
- **Additional features**: `[dEdx, 0, 0, 0, time]` where dEdx is preprocessed `gamma{1,2}_dEdx` and time is `gamma{1,2}_time / 10000`.

### Blob (PID 2) and Aggregated Blob (PID 6)

Source: `MasterAnaDev_Blob{X,Y,Z,T,TPos,TotalE}` branches.

Blobs are calorimetric energy deposits without a specific particle hypothesis. Their position `(X, Y, Z)` is converted to a pseudo-momentum by normalizing the direction vector and scaling by energy, treating them as massless particles originating from the origin.

- **Four-momentum** `(px, py, pz, E)`: constructed as `(X̂ · E, Ŷ · E, Ẑ · E, E)` where `(X̂, Ŷ, Ẑ)` is the unit direction vector from the origin.
- **Additional features**: `[0, x, y, z, time]` where coordinates come from `Blob{X,Y,Z} / 10000` and time from `BlobT / 10000`.

If there are more than 20 blobs, the top 19 by energy are kept individually and the rest are summed into a single aggregated blob (PID 6) whose additional features are averaged.

### Prong (PID 3, 4, 5) and Aggregated Prong (PID 7)

Source: `prong_part_{pos, E, score, mass, charge, pid}` and `prong_dEdXMean` branches.

Prongs are track-based reconstructed particle candidates. Each prong has a particle hypothesis assigned by the reconstruction:

| Original `prong_part_pid` | Assigned PID | Hypothesis |
|---|---|---|
| 3 | 3 | Pion |
| 8 | 4 | EM shower |
| 13 | 5 | Muon-like |

Prongs with `prong_part_pid = −999` or `0` are filtered out, as well as those with energy ≈ 0 (`prong_part_E[..., 3] < 1e-6`).

- **Four-momentum** `(px, py, pz, E)`: from `prong_part_E` (4 columns, treated as a four-vector).
- **Additional features**: `[dEdx, x, y, z, time]` where dEdx is preprocessed `prong_dEdXMean`, and coordinates/time come from `prong_part_pos / 10000` (4 columns: x, y, z, t).

If there are more than 10 prongs, the top 9 by energy are kept individually and the rest are summed into an aggregated prong (PID 7).

---

## Global Features (`global_features`)

A 13-dimensional vector per event providing event-level context.

### Columns 0–3: Calorimetric and muon-related

| Index | Name | Transform |
|---|---|---|
| 0 | Muon fuzz energy | `log(muon_fuzz_energy + 1e-5)` (negative values clipped to 0 before log) |
| 1 | Muon isolated blobs energy | `log(muon_iso_blobs_energy + 1e-5)` (negative values clipped to 0 before log) |
| 2 | Hadronic recoil energy | `log(MasterAnaDev_hadron_recoil + 1e-5)` (negative values clipped to 0 before log) |
| 3 | Number of Michel electrons | Raw integer count from `improved_nmichel` |

### Columns 4–6: Reconstruction summaries (from point-cloud inputs)

| Index | Name | Description |
|---|---|---|
| 4 | Reconstructed muon flag | `1` if at least one MINOS-matched muon after overflow cleaning, else `0` (same selection as the muon token in `data`) |
| 5 | γγ invariant mass | If exactly two reconstructed photons (`gamma1` / `gamma2` with `E > 1e-5`), invariant mass in MeV from `(px, py, pz, E)`; else `0` |
| 6 | Charged prong count | Number of prongs with `|prong_part_charge| > 1e-6` (after prong filtering) |

### Columns 7–12: Energy sums by particle type

These are the total energy deposited per node type, log-transformed: `log(Σ E + 1e-3)`, where E is computed as `exp(log_E_feature)` for all particles of that type in the event.

| Index | PID summed | Description |
|---|---|---|
| 7 | 2 | Total blob energy |
| 8 | 3 | Total prong (pion hypothesis) energy |
| 9 | 4 | Total prong (EM shower hypothesis) energy |
| 10 | 5 | Total prong (muon-like hypothesis) energy |
| 11 | 6 | Aggregated blob energy |
| 12 | 7 | Aggregated prong energy |

---

## Truth Labels (`truth_labels`)

A 15-dimensional vector per event containing Monte Carlo ground truth.

### Columns 0–10: Scalar labels

| Index | Name | Description |
|---|---|---|
| 0 | `mc_incomingE` | True incoming neutrino energy (MeV) |
| 1 | `mc_intType` | GENIE interaction type: 1=QE, 2=RES, 3=DIS, 4=COH, 8=MEC |
| 2 | `E_nu_true / E_mu_reco` | Ratio of true neutrino energy to reconstructed muon energy (−1 if muon is invalid) |
| 3 | `mc_current` | Current type: 1=CC (Charged Current), 2=NC (Neutral Current) |
| 4 | CC pion label | Single-pion production tag: 0=other, 1=CC with exactly one π⁺, 2=CC with exactly one π⁻ |
| 5 | `n_pi_plus` | Number of π⁺ in the MC final state |
| 6 | `n_pi_minus` | Number of π⁻ in the MC final state |
| 7 | `is_multi_pion` | 1 if more than one π⁺ or more than one π⁻ (legacy label, noted as potentially incorrect in code) |
| 8 | E_available (with muon) | Sum of available energy including muon: Σ T(baryons) + Σ E(mesons, leptons, photons) + E(muon) |
| 9 | E_available (no muon) | Same as above but excluding the muon contribution |
| 10 | `n_pi_zero` | Number of π⁰ in the MC final state |

**Available energy** is computed from MC final-state particles using PDG masses: kinetic energy T = E − m for baryons (protons, anti-protons), and total energy E for mesons (π±, π⁰, K±), leptons (e±), and photons. Negative values are clipped to 0.

### Columns 11–14: Pion four-vector

| Index | Name | Description |
|---|---|---|
| 11 | π px | MC truth pion px (nonzero only for CC single-pion events) |
| 12 | π py | MC truth pion py |
| 13 | π pz | MC truth pion pz |
| 14 | π E | MC truth pion energy |

---

## Directory Structure

After running `preprocess_dataset.py` followed by `split_dataset.py`, the output for a given playlist (e.g., 1A) looks like:

```
$DATA_DIR/Minerva/20260313/
├── 1A/
│   ├── train/
│   │   └── 0.pb
│   ├── val/
│   │   └── 0.pb
│   └── test/
│       └── 0.pb
├── 1B/
│   └── ...
└── result.pkl          # train/val/test index split metadata
```

---

## Preprocessing Pipeline Summary

1. **`preprocess_dataset.py`**: Reads raw ROOT files (MasterAnaDev TTree), extracts muon, photon, blob, and prong collections, converts four-momenta to (η, φ, log pT, log E), assigns PIDs, aggregates excess blobs/prongs, computes global features and truth labels, and saves each ROOT file as a `.pb` PyTorch nested tensor.

2. **`split_dataset.py`**: Loads all `.pb` files for a playlist, filters to allowed interaction types {1, 2, 3, 4, 8}, performs a stratified train/val/test split (by `mc_intType`), and saves the splits as single `.pb` files per set.

3. **`extract_baselines.py`** (optional): Computes traditional neutrino energy reconstruction baselines (CCQE formula, E_μ + E_recoil, etc.) from the raw ROOT files for comparison with ML predictions.
