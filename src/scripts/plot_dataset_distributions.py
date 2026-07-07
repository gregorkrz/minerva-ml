#!/usr/bin/env python3
"""Plot distributions of all variables in the ML-ready Minerva .pb dataset.

Reads preprocessed playlists (train/val/test ``*.pb`` files) and writes histograms for:
  - per-particle features (10 columns in ``data``)
  - per-event global features (16 columns)
  - per-event truth labels (15 columns)
  - particles-per-event multiplicity
  - poster figures in ``<out-dir>/poster/`` (token multiplicity, blob/prong
    multiplicity, blob/prong token energies, energy comparison, E_available
    spectrum, E_available by Pi_labels_v2 class)
  - alternate poster styling in ``<out-dir>/poster_v2/`` (two reds, Matter font)
  - optional JetClass-II poster figures and joint MINERvA/JetClass-II overlays

Example::

    python -m src.scripts.plot_dataset_distributions \\
        --data-path /pscratch/sd/g/gregork/Minerva/20260326/20260326 \\
        --playlist 1A --split train --out-dir plots/dataset_distributions

    # Quick check on a subsample::
    python -m src.scripts.plot_dataset_distributions --max-events 50000

    # Regenerate poster plots from cached bin values (no dataset read)::
    python -m src.scripts.plot_dataset_distributions \\
        --playlist 1A --split train --out-dir plots/dataset_distributions \\
        --poster-from-cache

    # JetClass-II poster figures (separate from MINERvA)::
    python -m src.scripts.plot_dataset_distributions \\
        --playlist 1A --split train --jetclass2-n-files 1 --max-events 50000
"""

import argparse
import json
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.dataset.dataloader import get_Pi_labels_v2

# Per-particle columns (DATASET.md)
PARTICLE_NAMES = [
    "eta",
    "phi",
    "log_pT",
    "log_E",
    "PID",
    "dEdx_log",
    "x_div1e4",
    "y_div1e4",
    "z_div1e4",
    "t_div1e4",
]
PARTICLE_DISCRETE = {4}  # PID

GLOBAL_NAMES = [
    "log_muon_fuzz_E",
    "log_muon_iso_blobs_E",
    "log_hadron_recoil",
    "log_passive_recoil_ID",
    "log_passive_recoil_OD",
    "log_passive_recoil_sum",
    "n_michel",
    "has_reco_muon",
    "gg_inv_mass_MeV",
    "charged_pion_prong_count",
    "log_sum_E_blob",
    "log_sum_E_prong_pion",
    "log_sum_E_prong_EM",
    "log_sum_E_prong_muonlike",
    "log_sum_E_agg_blob",
    "log_sum_E_agg_prong",
]
GLOBAL_DISCRETE = {6, 7, 9}

TRUTH_NAMES = [
    "mc_incomingE_MeV",
    "mc_intType",
    "E_nu_over_E_mu_reco",
    "mc_current",
    "CC_1pi_label",
    "n_pi_plus",
    "n_pi_minus",
    "is_multi_pion",
    "E_available_with_muon",
    "E_available_no_muon",
    "n_pi_zero",
    "pi_px",
    "pi_py",
    "pi_pz",
    "pi_E",
]
TRUTH_DISCRETE = {1, 3, 4, 5, 6, 7, 10}
# Column 2 uses -1 as invalid sentinel
TRUTH_SENTINEL = {2: -1.0}
E_AVAILABLE_WITH_MUON_COL = 8
E_AVAILABLE_NO_MUON_COL = 9

INT_TYPE_LABELS = {1: "QE", 2: "RES", 3: "DIS", 4: "COH", 8: "MEC"}
CURRENT_LABELS = {1: "CC", 2: "NC"}

POSTER_FIGSIZE = (4.5, 4.5)
POSTER_ENERGY_FIGSIZE = (POSTER_FIGSIZE[0] * 1.5, POSTER_FIGSIZE[1])
JOINT_POSTER_FIGSIZE = (
    POSTER_ENERGY_FIGSIZE[0] * 1.15,
    POSTER_ENERGY_FIGSIZE[1] * 0.8,
)
POSTER_ENERGY_XLIM = (0.0, 15.0)
POSTER_ENERGY_NBINS = 60
BLOB_PRONG_ENERGY_XMAX_MEV = 1.0e4  # 10 GeV
POSTER_PNG_DPI = 200
POSTER_V2_PNG_DPI = POSTER_PNG_DPI * 2
JOINT_ENERGY_XMAX_GEV = 1e4

JETCLASS2_DEFAULT_PATH = (
    "/global/cfs/cdirs/m3246/jaluus/data/omnilearned/jetclass2/train"
)
# Preprocessed OmniLearned / HyperScale h5 contract (see HyperScale DATASET.md):
#   data (N, 150, 9) float32 — cols 0–3: (Δη, Δφ, log pT, log E); cols 4–8: extra
#       features or zero-filled padding slots
#   mask (N, 150) bool — real-particle mask (top tagging split files)
#   pid (N,) int64 — event label
#   global (N, 3) float32 — optional jet-level features on sharded datasets
#       (JetClass-II: log jet energy, log jet pT, n_particles / 100; energies in GeV)
PREPROCESSED_LOG_E_COL = 3
PREPROCESSED_KIN_COLS = slice(0, 4)
PREPROCESSED_JET_ENERGY_COL = 0
MINERVA_COLOR = "#175e54"
JETCLASS2_COLOR = "#52b596"
JETCLASS2_SUM_E_COLOR = "#7ecbb3"
POSTER_V2_COLOR_PRIMARY = "#8B1A1A"
POSTER_V2_COLOR_SECONDARY = "#310000"
PARTICLE_PID_COL = 4
BLOB_PIDS = (2, 6)
PRONG_PIDS = (3, 4, 5, 7)

PI_LABELS_V2_NAMES = {
    0: r"CC $1\pi^\pm$",
    1: r"CC $N\pi^\pm$ ($N>1$)",
    2: r"CC $1\pi^0$",
    3: "CC other",
    4: "NC",
}
POSTER_CLASS_COLORS = [
    MINERVA_COLOR,
    "#2a8a7a",
    "#52b596",
    "#7ecbb3",
    "#a8d5c8",
]
POSTER_V2_CLASS_COLORS = [
    POSTER_V2_COLOR_PRIMARY,
    "#C44D4D",
    POSTER_V2_COLOR_SECONDARY,
    "#6B2020",
    "#1A0000",
]
E_AVAILABLE_PLOT_XLIM = (0.0, 30.0)
# Distinct palettes for e_available_by_class (6 series incl. E_nu).
POSTER_BY_CLASS_COLORS = [
    "#175e54",  # CC 1pi
    "#d95f02",  # CC Npi
    "#1f77b4",  # CC 1pi0
    "#9467bd",  # CC other
    "#8c564b",  # NC
]
POSTER_V2_BY_CLASS_COLORS = [
    "#8B1A1A",  # CC 1pi
    "#2E5FA8",  # CC Npi
    "#C4841A",  # CC 1pi0
    "#4A154B",  # CC other
    "#2D6A4F",  # NC
]
E_NU_BY_CLASS_COLOR = "#3d3d3d"
E_NU_BY_CLASS_V2_COLOR = "#1a1a1a"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-path",
        type=str,
        default="/pscratch/sd/g/gregork/Minerva/20260326/20260326",
        help="Root with playlist subdirs (1A/, 1B/, ...)",
    )
    p.add_argument("--playlist", type=str, default="1A", help="Playlist folder name")
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split subdirectory to read",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="plots/dataset_distributions",
        help="Output directory for PDFs and summary JSON",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Subsample at most this many events (for faster runs)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for subsampling")
    p.add_argument(
        "--bins",
        type=int,
        default=80,
        help="Histogram bins for continuous variables",
    )
    p.add_argument(
        "--log-y",
        action="store_true",
        help="Use log scale on y-axis for continuous histograms",
    )
    p.add_argument(
        "--poster-from-cache",
        action="store_true",
        help="Regenerate poster plots from <out-dir>/poster/poster_bins_<tag>.pkl only",
    )
    p.add_argument(
        "--jetclass2-path",
        type=str,
        default=JETCLASS2_DEFAULT_PATH,
        help=(
            "Preprocessed h5 path: directory of sharded JetClass2_*.h5 files, "
            "or a single split file (e.g. train_top_tagging.h5)"
        ),
    )
    p.add_argument(
        "--jetclass2-n-files",
        type=int,
        default=1,
        help="Number of JetClass2_*.h5 files to sample (random, without replacement)",
    )
    p.add_argument(
        "--no-jetclass2",
        action="store_true",
        help="Skip JetClass-II overlay on poster plots",
    )
    return p.parse_args()


def _subsample_from_nested(folder, max_events, seed):
    paths = sorted(folder.glob("*.pb"))
    if not paths:
        raise FileNotFoundError(f"No .pb files in {folder}")

    rng = np.random.default_rng(seed)
    particle_rows = []
    global_list = []
    truth_list = []
    n_pt_list = []
    sum_e_list = []
    n_features = 10

    for path in paths:
        # Nested jagged tensors require full unpickling (trusted local .pb files).
        blob = torch.load(path, map_location="cpu", weights_only=False)
        data = blob["data"]
        global_f = np.asarray(blob["global_features"], dtype=np.float64)
        truth = np.asarray(blob["truth_labels"], dtype=np.float64)
        offsets = data.offsets()
        n_ev = len(offsets) - 1
        values = np.asarray(data.values(), dtype=np.float64)
        n_features = values.shape[1]

        event_indices = np.arange(n_ev)
        if max_events is not None:
            remaining = max_events - len(n_pt_list)
            if remaining <= 0:
                break
            if remaining < n_ev:
                event_indices = np.sort(rng.choice(n_ev, size=remaining, replace=False))

        for i in event_indices:
            start = int(offsets[i].item())
            end = int(offsets[i + 1].item())
            n_pt = end - start
            n_pt_list.append(n_pt)
            if n_pt > 0:
                rows = values[start:end]
                particle_rows.append(rows)
                sum_e_list.append(float(np.sum(np.exp(rows[:, 3]))))
            else:
                sum_e_list.append(0.0)
            global_list.append(global_f[i])
            truth_list.append(truth[i])

        if max_events is not None and len(n_pt_list) >= max_events:
            break

    particles = (
        np.concatenate(particle_rows, axis=0)
        if particle_rows
        else np.zeros((0, n_features), dtype=np.float64)
    )
    global_features = np.stack(global_list, axis=0)
    truth_labels = np.stack(truth_list, axis=0)
    n_particles_per_event = np.asarray(n_pt_list, dtype=np.int32)
    sum_token_energy_per_event = np.asarray(sum_e_list, dtype=np.float64)
    return (
        particles,
        global_features,
        truth_labels,
        n_particles_per_event,
        sum_token_energy_per_event,
    )


def _select_h5_files(folder: Path, n_files: int, seed: int) -> list[Path]:
    paths = sorted(folder.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"No .h5 files in {folder}")
    if n_files <= 0:
        raise ValueError("--jetclass2-n-files must be >= 1")
    if n_files >= len(paths):
        return paths
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(paths), size=n_files, replace=False))
    return [paths[int(i)] for i in indices]


def _resolve_h5_inputs(path: Path, n_files: int, seed: int) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"JetClass-II path not found: {path}")
    return _select_h5_files(path, n_files, seed)


def _particle_mask(data: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Real-particle mask following the pretraining batch contract."""
    if mask is not None:
        return mask.astype(bool)
    # Sharded OmniLearned files omit mask; padding keeps kinematics zero-filled.
    return np.abs(data[:, :, PREPROCESSED_KIN_COLS]).sum(axis=2) > 0.0


def _load_preprocessed_h5_file(
    path: Path,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return n_particles/event, optional reference energy (GeV), sum token E (GeV)."""
    with h5py.File(path, "r") as f:
        data = np.asarray(f["data"], dtype=np.float64)
        mask = np.asarray(f["mask"], dtype=bool) if "mask" in f else None
        global_f = np.asarray(f["global"], dtype=np.float64) if "global" in f else None

    pmask = _particle_mask(data, mask)
    log_e = data[:, :, PREPROCESSED_LOG_E_COL]
    n_per_event = pmask.sum(axis=1).astype(np.int32)
    sum_token_e = np.where(pmask, np.exp(log_e), 0.0).sum(axis=1)

    ref_energy = None
    if global_f is not None:
        ref_energy = np.exp(global_f[:, PREPROCESSED_JET_ENERGY_COL])

    return n_per_event, ref_energy, sum_token_e


def _load_jetclass2_sample(
    path: Path, n_files: int, seed: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, list[str]]:
    """Load poster quantities from preprocessed h5 file(s)."""
    chosen = _resolve_h5_inputs(path, n_files, seed)
    n_pt_chunks: list[np.ndarray] = []
    ref_e_chunks: list[np.ndarray] = []
    sum_e_chunks: list[np.ndarray] = []
    ref_present = True

    for h5_path in chosen:
        n_per_event, ref_energy, sum_token_e = _load_preprocessed_h5_file(h5_path)
        n_pt_chunks.append(n_per_event)
        sum_e_chunks.append(sum_token_e)
        if ref_energy is None:
            ref_present = False
        else:
            ref_e_chunks.append(ref_energy)

    ref_out = np.concatenate(ref_e_chunks) if ref_present else None
    return (
        np.concatenate(n_pt_chunks),
        ref_out,
        np.concatenate(sum_e_chunks),
        [str(p) for p in chosen],
    )


def _finite_mask(values, sentinel=None):
    m = np.isfinite(values)
    if sentinel is not None:
        m &= values != sentinel
    return m


def _summary(values, sentinel=None):
    m = _finite_mask(values, sentinel)
    if not np.any(m):
        return {"count": int(values.size), "valid": 0}
    v = values[m]
    return {
        "count": int(values.size),
        "valid": int(v.size),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
    }


def _plot_discrete(ax, values, title, xlabels=None):
    uniq, counts = np.unique(values.astype(np.int64), return_counts=True)
    order = np.argsort(uniq)
    uniq, counts = uniq[order], counts[order]
    ax.bar(uniq.astype(float), counts, width=0.85, color="steelblue", edgecolor="white")
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("count")
    if xlabels:
        ticks = [int(u) for u in uniq if int(u) in xlabels]
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([xlabels[t] for t in ticks], rotation=35, ha="right")
    ax.tick_params(labelsize=7)


def _plot_continuous(ax, values, title, bins: int, log_y: bool, sentinel=None):
    m = _finite_mask(values, sentinel)
    v = values[m]
    ax.hist(v, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("count")
    if log_y and len(v):
        ax.set_yscale("log")
    ax.tick_params(labelsize=7)
    if len(v):
        ax.text(
            0.98,
            0.97,
            f"n={len(v):,}\nmed={np.median(v):.4g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )


def _plot_grid(
    values_2d: np.ndarray,
    names: list[str],
    discrete_cols: set[int],
    out_path: Path,
    title: str,
    bins: int,
    log_y: bool,
    discrete_xlabels=None,
    sentinels=None,
):
    n_cols = values_2d.shape[1]
    n_rows = int(np.ceil(n_cols / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(14, 3.2 * n_rows))
    axes = np.atleast_2d(axes)
    summaries = {}
    sentinels = sentinels or {}
    xlabels_map = discrete_xlabels or {}

    for j in range(n_rows * 4):
        r, c = divmod(j, 4)
        ax = axes[r, c]
        if j >= n_cols:
            ax.axis("off")
            continue
        col = values_2d[:, j]
        name = names[j]
        sent = sentinels.get(j)
        summaries[name] = _summary(col, sent)
        if j in discrete_cols:
            _plot_discrete(ax, col, name, xlabels_map.get(j))
        else:
            _plot_continuous(ax, col, name, bins, log_y, sent)
    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return summaries


def _poster_style():
    return {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
    }


def _poster_v2_style():
    return {**_poster_style(), "font.family": "Matter"}


def _poster_cache_path(poster_dir: Path, tag: str) -> Path:
    return poster_dir / f"poster_bins_{tag}.pkl"


def _jetclass2_poster_cache_path(poster_dir: Path, tag: str) -> Path:
    return poster_dir / f"poster_bins_jetclass2_{tag}.pkl"


def _jetclass2_poster_section(
    n_per_event: np.ndarray,
    jet_energy_gev: np.ndarray | None,
    sum_token_energy_gev: np.ndarray,
    meta: dict,
) -> dict:
    token_bin_edges = np.arange(0, max(int(n_per_event.max()) + 2, 2))
    token_counts, _ = np.histogram(n_per_event, bins=token_bin_edges)
    sum_m = _finite_mask(sum_token_energy_gev) & (sum_token_energy_gev > 0)
    sum_e = sum_token_energy_gev[sum_m]
    jet_e = None
    if jet_energy_gev is not None:
        jet_m = _finite_mask(jet_energy_gev) & (jet_energy_gev > 0)
        jet_e = jet_energy_gev[jet_m]

    ref_energies = sum_e if jet_e is None else np.concatenate([sum_e, jet_e])
    xlim = (
        0.0,
        float(np.percentile(ref_energies, 99) * 1.05) if ref_energies.size else 500.0,
    )
    energy_bin_edges = np.linspace(xlim[0], xlim[1], POSTER_ENERGY_NBINS + 1)
    sum_counts, _ = np.histogram(sum_e, bins=energy_bin_edges)

    energy_series = [
        {
            "counts_key": "sum_token_counts",
            "label": "JetClass2: Sum of token energies",
            "color": JETCLASS2_SUM_E_COLOR,
        },
    ]
    energy_bins = {
        "bin_edges": energy_bin_edges,
        "sum_token_counts": sum_counts,
        "xlim": xlim,
        "series": energy_series,
    }
    if jet_e is not None:
        jet_counts, _ = np.histogram(jet_e, bins=energy_bin_edges)
        energy_bins["jet_energy_counts"] = jet_counts
        energy_series.insert(
            0,
            {
                "counts_key": "jet_energy_counts",
                "label": "JetClass2: Jet energy",
                "color": JETCLASS2_COLOR,
            },
        )

    return {
        "meta": meta,
        "tokens_per_event": {
            "bin_edges": token_bin_edges,
            "counts": token_counts,
            "log_y": True,
            "color": JETCLASS2_COLOR,
        },
        "energy_comparison": energy_bins,
    }


def _save_jetclass2_poster(
    jetclass2_cache: dict, poster_dir: Path, tag: str
) -> Path | None:
    jc_cache_path = _jetclass2_poster_cache_path(poster_dir, tag)
    _write_jetclass2_poster_plots(jetclass2_cache, poster_dir)
    if _save_poster_cache(jetclass2_cache, jc_cache_path):
        return jc_cache_path
    return None


def _load_or_build_jetclass2_poster(
    args, poster_dir: Path, tag: str
) -> tuple[dict | None, Path | None]:
    if args.no_jetclass2:
        return None, None

    jc_cache_path = _jetclass2_poster_cache_path(poster_dir, tag)
    if jc_cache_path.exists():
        try:
            cached = _load_poster_cache(jc_cache_path)
        except Exception as exc:
            print(f"JetClass-II cache unreadable ({exc}); rebuilding ...")
            cached = None
        if cached is not None:
            meta = cached.get("meta", {})
            if (
                meta.get("n_files") == args.jetclass2_n_files
                and meta.get("seed") == args.seed
                and meta.get("data_path") == str(Path(args.jetclass2_path))
            ):
                print(f"Loading JetClass-II poster cache {jc_cache_path} ...")
                return cached, jc_cache_path
            print(
                "JetClass-II cache stale "
                f"(n_files={meta.get('n_files')} vs {args.jetclass2_n_files}); rebuilding ..."
            )

    jc_folder = Path(args.jetclass2_path)
    print(
        f"Loading JetClass-II sample from {jc_folder} "
        f"({args.jetclass2_n_files} file(s), seed={args.seed}) ..."
    )
    jc_n, jc_jet_e, jc_sum_e, jc_files = _load_jetclass2_sample(
        jc_folder, args.jetclass2_n_files, args.seed
    )
    print(
        f"  JetClass-II events={len(jc_n):,}  "
        f"tokens/event: mean={jc_n.mean():.2f} max={jc_n.max()}"
    )
    jc_cache = _build_jetclass2_poster_cache(
        jc_n,
        jc_jet_e,
        jc_sum_e,
        tag,
        meta={
            "data_path": str(jc_folder),
            "files": jc_files,
            "n_files": args.jetclass2_n_files,
            "n_events": int(len(jc_n)),
            "seed": args.seed,
        },
    )
    jc_cache_path = _save_jetclass2_poster(jc_cache, poster_dir, tag)
    return jc_cache, jc_cache_path


def _pi_labels_v2_from_truth(truth_labels: np.ndarray) -> np.ndarray:
    labels = get_Pi_labels_v2(torch.from_numpy(truth_labels))
    return labels.numpy().astype(np.int64)


def _blob_prong_counts_per_event(
    particles: np.ndarray, n_per_event: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-event blob and prong token counts from reconstruction PID column."""
    pid = particles[:, PARTICLE_PID_COL].astype(np.int64)
    blob_counts = np.zeros(len(n_per_event), dtype=np.int32)
    prong_counts = np.zeros(len(n_per_event), dtype=np.int32)
    offset = 0
    for i, n in enumerate(n_per_event):
        n = int(n)
        if n > 0:
            pids = pid[offset : offset + n]
            blob_counts[i] = int(np.isin(pids, BLOB_PIDS).sum())
            prong_counts[i] = int(np.isin(pids, PRONG_PIDS).sum())
        offset += n
    return blob_counts, prong_counts


def _build_blob_prong_multiplicity_bins(
    blob_counts: np.ndarray, prong_counts: np.ndarray
) -> dict:
    """Binned blob and prong multiplicity per event."""
    hi = max(int(blob_counts.max()), int(prong_counts.max()), 1) + 1
    bin_edges = np.arange(0, hi + 1, dtype=np.float64)
    blob_hist, _ = np.histogram(blob_counts, bins=bin_edges)
    prong_hist, _ = np.histogram(prong_counts, bins=bin_edges)
    return {
        "bin_edges": bin_edges,
        "blob_counts": blob_hist,
        "prong_counts": prong_hist,
        "log_y": True,
        "xlim": (0.0, float(hi)),
        "series": [
            {
                "counts_key": "blob_counts",
                "label": "Blobs per event",
                "color": MINERVA_COLOR,
            },
            {
                "counts_key": "prong_counts",
                "label": "Prongs per event",
                "color": "#2a8a7a",
            },
        ],
    }


def _blob_prong_token_energies_mev(
    particles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-token energies (MeV) for blob and prong reconstruction objects."""
    pid = particles[:, PARTICLE_PID_COL].astype(np.int64)
    e_mev = np.exp(particles[:, 3])
    return e_mev[np.isin(pid, BLOB_PIDS)], e_mev[np.isin(pid, PRONG_PIDS)]


def _build_blob_prong_energy_bins(
    blob_e_mev: np.ndarray, prong_e_mev: np.ndarray
) -> dict:
    """Binned blob and prong token energy spectra (MeV, log-spaced bins)."""
    blob_e = np.asarray(blob_e_mev, dtype=np.float64)
    prong_e = np.asarray(prong_e_mev, dtype=np.float64)
    blob_m = _finite_mask(blob_e) & (blob_e > 0)
    prong_m = _finite_mask(prong_e) & (prong_e > 0)
    ref = (
        np.concatenate([blob_e[blob_m], prong_e[prong_m]])
        if np.any(blob_m) or np.any(prong_m)
        else np.array([], dtype=np.float64)
    )
    lo, hi = 0.1, BLOB_PRONG_ENERGY_XMAX_MEV
    if ref.size:
        lo = max(float(np.percentile(ref, 0.1)), 0.01)
        lo = min(lo, hi / 1e3)
    bin_edges = np.logspace(np.log10(lo), np.log10(hi), POSTER_ENERGY_NBINS + 1)
    blob_counts, _ = np.histogram(blob_e[blob_m], bins=bin_edges)
    prong_counts, _ = np.histogram(prong_e[prong_m], bins=bin_edges)
    return {
        "bin_edges": bin_edges,
        "blob_energy_counts": blob_counts,
        "prong_energy_counts": prong_counts,
        "xlim": (lo, hi),
        "log_x": True,
        "log_y": True,
        "xlabel": "Energy [MeV]",
        "ylabel": "Tokens (arb. scale)",
        "series": [
            {
                "counts_key": "blob_energy_counts",
                "label": "Blob energy",
                "color": MINERVA_COLOR,
            },
            {
                "counts_key": "prong_energy_counts",
                "label": "Prong energy",
                "color": "#2a8a7a",
            },
        ],
    }


def _e_available_xlim(e_gev: np.ndarray) -> tuple[float, float]:
    valid = _finite_mask(e_gev) & (e_gev >= 0)
    positives = e_gev[valid]
    xlim = POSTER_ENERGY_XLIM
    if positives.size:
        hi = float(np.percentile(positives, 99) * 1.05)
        xlim = (0.0, max(hi, POSTER_ENERGY_XLIM[1]))
    return xlim


def _build_e_available_spectrum_bins(
    e_avail_with_muon_mev: np.ndarray,
    e_avail_no_muon_mev: np.ndarray,
) -> dict:
    """Binned MC E_available spectra (with / without muon), in GeV."""
    with_gev = np.asarray(e_avail_with_muon_mev, dtype=np.float64) / 1000.0
    no_gev = np.asarray(e_avail_no_muon_mev, dtype=np.float64) / 1000.0
    with_m = _finite_mask(with_gev) & (with_gev >= 0)
    no_m = _finite_mask(no_gev) & (no_gev >= 0)
    xlim = E_AVAILABLE_PLOT_XLIM
    bin_edges = np.linspace(xlim[0], xlim[1], POSTER_ENERGY_NBINS + 1)
    with_counts, _ = np.histogram(with_gev[with_m], bins=bin_edges)
    no_counts, _ = np.histogram(no_gev[no_m], bins=bin_edges)
    return {
        "bin_edges": bin_edges,
        "with_muon_counts": with_counts,
        "no_muon_counts": no_counts,
        "xlim": xlim,
        "log_y": True,
        "series": [
            {
                "counts_key": "no_muon_counts",
                "label": r"$E_\mathrm{available}$",
                "color": MINERVA_COLOR,
            },
            {
                "counts_key": "with_muon_counts",
                "label": r"$E_\mathrm{available} + E_\mu$",
                "color": "#2a8a7a",
            },
        ],
    }


def _build_e_available_by_class_bins(
    e_avail_no_muon_mev: np.ndarray,
    pi_labels: np.ndarray,
    incoming_e_mev: np.ndarray | None = None,
) -> dict:
    """Binned E_available (no muon) spectrum per Pi_labels_v2 class, in GeV."""
    e_gev = np.asarray(e_avail_no_muon_mev, dtype=np.float64) / 1000.0
    pi_labels = np.asarray(pi_labels, dtype=np.int64)
    valid = _finite_mask(e_gev) & (e_gev >= 0)
    xlim = E_AVAILABLE_PLOT_XLIM
    bin_edges = np.linspace(xlim[0], xlim[1], POSTER_ENERGY_NBINS + 1)
    bins: dict = {
        "bin_edges": bin_edges,
        "xlim": xlim,
        "log_y": True,
        "xlabel": "Energy [GeV]",
        "series": [],
    }
    if incoming_e_mev is not None:
        e_nu_gev = np.asarray(incoming_e_mev, dtype=np.float64) / 1000.0
        e_nu_valid = _finite_mask(e_nu_gev) & (e_nu_gev > 0)
        e_nu_counts, _ = np.histogram(e_nu_gev[e_nu_valid], bins=bin_edges)
        bins["e_nu_counts"] = e_nu_counts
        bins["series"].append(
            {
                "counts_key": "e_nu_counts",
                "label": r"$E_\nu$",
                "color": E_NU_BY_CLASS_COLOR,
                "v2_color": E_NU_BY_CLASS_V2_COLOR,
            }
        )
    for cls in range(5):
        mask = valid & (pi_labels == cls)
        counts, _ = np.histogram(e_gev[mask], bins=bin_edges)
        counts_key = f"class_{cls}_counts"
        bins[counts_key] = counts
        bins["series"].append(
            {
                "counts_key": counts_key,
                "label": PI_LABELS_V2_NAMES[cls],
                "color": POSTER_BY_CLASS_COLORS[cls],
                "v2_color": POSTER_V2_BY_CLASS_COLORS[cls],
            }
        )
    return bins


def _build_poster_cache(
    n_per_event: np.ndarray,
    incoming_e_mev: np.ndarray,
    sum_token_e_mev: np.ndarray,
    e_avail_with_muon_mev: np.ndarray,
    e_avail_no_muon_mev: np.ndarray,
    truth_labels: np.ndarray,
    blob_counts: np.ndarray,
    prong_counts: np.ndarray,
    blob_e_mev: np.ndarray,
    prong_e_mev: np.ndarray,
    tag: str,
    meta: dict,
) -> dict:
    pi_labels = _pi_labels_v2_from_truth(truth_labels)
    token_bin_edges = np.arange(0, max(int(n_per_event.max()) + 2, 2))
    token_counts, _ = np.histogram(n_per_event, bins=token_bin_edges)

    energy_bin_edges = np.linspace(
        POSTER_ENERGY_XLIM[0],
        POSTER_ENERGY_XLIM[1],
        POSTER_ENERGY_NBINS + 1,
    )
    incoming_gev = incoming_e_mev / 1000.0
    sum_token_gev = sum_token_e_mev / 1000.0
    incoming_m = _finite_mask(incoming_gev)
    sum_m = _finite_mask(sum_token_gev) & (sum_token_gev > 0)
    incoming_counts, _ = np.histogram(incoming_gev[incoming_m], bins=energy_bin_edges)
    sum_token_counts, _ = np.histogram(sum_token_gev[sum_m], bins=energy_bin_edges)

    cache = {
        "tag": tag,
        "meta": meta,
        "tokens_per_event": {
            "bin_edges": token_bin_edges,
            "counts": token_counts,
            "log_y": True,
            "color": MINERVA_COLOR,
        },
        "blob_prong_multiplicity": _build_blob_prong_multiplicity_bins(
            blob_counts,
            prong_counts,
        ),
        "blob_prong_energy": _build_blob_prong_energy_bins(blob_e_mev, prong_e_mev),
        "energy_comparison": {
            "bin_edges": energy_bin_edges,
            "incoming_counts": incoming_counts,
            "sum_token_counts": sum_token_counts,
            "xlim": POSTER_ENERGY_XLIM,
            "series": [
                {
                    "counts_key": "incoming_counts",
                    "label": "MINERvA: Neutrino energy",
                    "color": MINERVA_COLOR,
                },
                {
                    "counts_key": "sum_token_counts",
                    "label": "Sum of token energies",
                    "color": "#2a8a7a",
                },
            ],
        },
        "e_available_spectrum": _build_e_available_spectrum_bins(
            e_avail_with_muon_mev,
            e_avail_no_muon_mev,
        ),
        "e_available_by_class": _build_e_available_by_class_bins(
            e_avail_no_muon_mev,
            pi_labels,
            incoming_e_mev,
        ),
        "incoming_e_gev": incoming_gev,
        "e_avail_with_muon_gev": e_avail_with_muon_mev / 1000.0,
        "e_avail_no_muon_gev": e_avail_no_muon_mev / 1000.0,
    }

    return cache


def _build_jetclass2_poster_cache(
    n_per_event: np.ndarray,
    jet_energy_gev: np.ndarray | None,
    sum_token_energy_gev: np.ndarray,
    tag: str,
    meta: dict,
) -> dict:
    section = _jetclass2_poster_section(
        n_per_event, jet_energy_gev, sum_token_energy_gev, meta
    )
    cache = {"tag": f"jetclass2_{tag}", "meta": meta, **section}
    if jet_energy_gev is not None:
        cache["jet_energy_gev"] = jet_energy_gev
    return cache


# Raw per-event arrays kept in memory for joint plots; omit from pickle (saves ~10+ MB).
_POSTER_CACHE_OMIT_KEYS = frozenset(
    {
        "jet_energy_gev",
        "incoming_e_gev",
        "e_avail_with_muon_gev",
        "e_avail_no_muon_gev",
    }
)


def _save_poster_cache(cache: dict, cache_path: Path) -> bool:
    slim = {k: v for k, v in cache.items() if k not in _POSTER_CACHE_OMIT_KEYS}
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(slim, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except OSError as exc:
        print(f"WARNING: could not write poster cache {cache_path}: {exc}")
        return False


def _load_poster_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        raise FileNotFoundError(f"Poster cache not found: {cache_path}")
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _save_poster_fig(fig, out_path: Path, png_dpi: int = POSTER_PNG_DPI):
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")


def _counts_to_density(counts: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Normalize histogram counts to a probability density (integral = 1)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return counts
    widths = np.diff(bin_edges)
    return counts / (total * widths)


def _poster_v2_series_colors() -> tuple[str, str]:
    return POSTER_V2_COLOR_PRIMARY, POSTER_V2_COLOR_SECONDARY


def _resolve_incoming_e_gev(minerva_cache: dict, args) -> np.ndarray:
    if "incoming_e_gev" in minerva_cache:
        return np.asarray(minerva_cache["incoming_e_gev"], dtype=np.float64)
    meta = minerva_cache["meta"]
    folder = Path(meta["data_path"])
    _, _, truth_labels, _, _ = _subsample_from_nested(
        folder, meta.get("max_events"), meta.get("seed", args.seed)
    )
    return truth_labels[:, 0] / 1000.0


def _resolve_e_available_gev(
    minerva_cache: dict, args
) -> tuple[np.ndarray, np.ndarray]:
    if (
        "e_avail_with_muon_gev" in minerva_cache
        and "e_avail_no_muon_gev" in minerva_cache
    ):
        return (
            np.asarray(minerva_cache["e_avail_with_muon_gev"], dtype=np.float64),
            np.asarray(minerva_cache["e_avail_no_muon_gev"], dtype=np.float64),
        )
    meta = minerva_cache["meta"]
    folder = Path(meta["data_path"])
    _, _, truth_labels, _, _ = _subsample_from_nested(
        folder, meta.get("max_events"), meta.get("seed", args.seed)
    )
    return (
        truth_labels[:, E_AVAILABLE_WITH_MUON_COL] / 1000.0,
        truth_labels[:, E_AVAILABLE_NO_MUON_COL] / 1000.0,
    )


def _resolve_truth_labels(minerva_cache: dict, args) -> np.ndarray:
    meta = minerva_cache["meta"]
    folder = Path(meta["data_path"])
    _, _, truth_labels, _, _ = _subsample_from_nested(
        folder, meta.get("max_events"), meta.get("seed", args.seed)
    )
    return truth_labels


def _resolve_particles_and_n_per_event(
    minerva_cache: dict, args
) -> tuple[np.ndarray, np.ndarray]:
    meta = minerva_cache["meta"]
    folder = Path(meta["data_path"])
    particles, _, _, n_per_event, _ = _subsample_from_nested(
        folder, meta.get("max_events"), meta.get("seed", args.seed)
    )
    return particles, n_per_event


def _resolve_blob_prong_multiplicity_bins(minerva_cache: dict, args) -> dict | None:
    if "blob_prong_multiplicity" in minerva_cache:
        return minerva_cache["blob_prong_multiplicity"]
    particles, n_per_event = _resolve_particles_and_n_per_event(minerva_cache, args)
    if n_per_event.size == 0:
        return None
    blob_counts, prong_counts = _blob_prong_counts_per_event(particles, n_per_event)
    return _build_blob_prong_multiplicity_bins(blob_counts, prong_counts)


def _resolve_blob_prong_energy_bins(minerva_cache: dict, args) -> dict | None:
    if "blob_prong_energy" in minerva_cache:
        return minerva_cache["blob_prong_energy"]
    particles, _ = _resolve_particles_and_n_per_event(minerva_cache, args)
    if particles.size == 0:
        return None
    blob_e, prong_e = _blob_prong_token_energies_mev(particles)
    return _build_blob_prong_energy_bins(blob_e, prong_e)


def _resolve_e_available_spectrum_bins(minerva_cache: dict, args) -> dict | None:
    if "e_available_spectrum" in minerva_cache:
        return minerva_cache["e_available_spectrum"]
    with_gev, no_gev = _resolve_e_available_gev(minerva_cache, args)
    if with_gev.size == 0 and no_gev.size == 0:
        return None
    return _build_e_available_spectrum_bins(with_gev * 1000.0, no_gev * 1000.0)


def _resolve_e_available_by_class_bins(minerva_cache: dict, args) -> dict | None:
    if "e_available_by_class" in minerva_cache:
        return minerva_cache["e_available_by_class"]
    _, no_gev = _resolve_e_available_gev(minerva_cache, args)
    if no_gev.size == 0:
        return None
    truth_labels = _resolve_truth_labels(minerva_cache, args)
    pi_labels = _pi_labels_v2_from_truth(truth_labels)
    return _build_e_available_by_class_bins(
        no_gev * 1000.0,
        pi_labels,
        truth_labels[:, 0],
    )


def _resolve_jet_energy_gev(jetclass2_cache: dict, args) -> np.ndarray | None:
    if "jet_energy_gev" in jetclass2_cache:
        return np.asarray(jetclass2_cache["jet_energy_gev"], dtype=np.float64)
    meta = jetclass2_cache["meta"]
    _, jet_e, _, _ = _load_jetclass2_sample(
        Path(meta["data_path"]),
        meta.get("n_files", args.jetclass2_n_files),
        meta.get("seed", args.seed),
    )
    return jet_e


def _build_joint_reference_energy_bins(
    incoming_gev: np.ndarray, jet_energy_gev: np.ndarray
) -> dict:
    incoming = incoming_gev[_finite_mask(incoming_gev) & (incoming_gev > 0)]
    jet_e = jet_energy_gev[_finite_mask(jet_energy_gev) & (jet_energy_gev > 0)]
    positives = (
        np.concatenate([incoming, jet_e])
        if incoming.size and jet_e.size
        else incoming if incoming.size else jet_e
    )
    lo = max(float(positives.min()), 1e-3) if positives.size else 1e-3
    hi = JOINT_ENERGY_XMAX_GEV
    edges = np.logspace(np.log10(lo), np.log10(hi), POSTER_ENERGY_NBINS + 1)
    incoming_counts, _ = np.histogram(incoming, bins=edges)
    jet_counts, _ = np.histogram(jet_e, bins=edges)
    return {
        "bin_edges": edges,
        "incoming_counts": incoming_counts,
        "jet_energy_counts": jet_counts,
        "xlim": (lo, hi),
    }


def _plot_poster_tokens_per_event(bins: dict, out_path: Path):
    """Step histogram of tokens per event from cached bin values."""
    with plt.rc_context(_poster_style()):
        fig, ax = plt.subplots(figsize=POSTER_FIGSIZE)
        edges = bins["bin_edges"]
        ax.stairs(
            _counts_to_density(bins["counts"], edges),
            edges,
            color=bins.get("color", MINERVA_COLOR),
            linewidth=2.2,
        )
        if bins.get("log_y", True):
            ax.set_yscale("log")
        ax.set_xlabel("Tokens per event / jet")
        ax.set_ylabel("Events (arb. scale)")
        ax.grid(True, which="both", axis="y", alpha=0.25, linestyle="--")
        ax.minorticks_off()
        fig.tight_layout()
        _save_poster_fig(fig, out_path)
        plt.close(fig)


def _plot_poster_energy_comparison(bins: dict, out_path: Path):
    """Overlaid step histograms from cached bin values."""
    with plt.rc_context(_poster_style()):
        fig, ax = plt.subplots(figsize=POSTER_ENERGY_FIGSIZE)
        edges = bins["bin_edges"]
        for series in bins["series"]:
            ax.stairs(
                _counts_to_density(bins[series["counts_key"]], edges),
                edges,
                color=series["color"],
                label=series["label"],
                linewidth=2.2,
            )
        xlim = bins.get("xlim", POSTER_ENERGY_XLIM)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel("Events (arb. scale)")
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            framealpha=1.0,
        )
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        fig.tight_layout()
        _save_poster_fig(fig, out_path)
        plt.close(fig)


def _plot_poster_tokens_per_event_v2(bins: dict, out_path: Path):
    """poster_v2: primary red histogram."""
    primary, _ = _poster_v2_series_colors()
    with plt.rc_context(_poster_v2_style()):
        fig, ax = plt.subplots(figsize=POSTER_FIGSIZE)
        edges = bins["bin_edges"]
        ax.stairs(
            _counts_to_density(bins["counts"], edges),
            edges,
            color=primary,
            linewidth=2.2,
        )
        if bins.get("log_y", True):
            ax.set_yscale("log")
        ax.set_xlabel("Tokens per event / jet")
        ax.set_ylabel("Events (arb. scale)")
        ax.grid(True, which="both", axis="y", alpha=0.25, linestyle="--")
        ax.minorticks_off()
        fig.tight_layout()
        _save_poster_fig(fig, out_path, png_dpi=POSTER_V2_PNG_DPI)
        plt.close(fig)


def _plot_poster_energy_comparison_v2(bins: dict, out_path: Path):
    """poster_v2: two red series."""
    primary, secondary = _poster_v2_series_colors()
    v2_colors = [primary, secondary]
    with plt.rc_context(_poster_v2_style()):
        fig, ax = plt.subplots(figsize=POSTER_ENERGY_FIGSIZE)
        edges = bins["bin_edges"]
        for idx, series in enumerate(bins["series"]):
            ax.stairs(
                _counts_to_density(bins[series["counts_key"]], edges),
                edges,
                color=v2_colors[idx % len(v2_colors)],
                label=series["label"],
                linewidth=2.2,
            )
        xlim = bins.get("xlim", POSTER_ENERGY_XLIM)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel("Events (arb. scale)")
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            framealpha=1.0,
        )
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        fig.tight_layout()
        _save_poster_fig(fig, out_path, png_dpi=POSTER_V2_PNG_DPI)
        plt.close(fig)


def _plot_poster_blob_prong_energy(bins: dict, out_path: Path):
    """Overlaid blob and prong token energy spectra (log-log, MeV)."""
    with plt.rc_context(_poster_style()):
        fig, ax = plt.subplots(figsize=POSTER_ENERGY_FIGSIZE)
        edges = bins["bin_edges"]
        for series in bins["series"]:
            ax.stairs(
                _counts_to_density(bins[series["counts_key"]], edges),
                edges,
                color=series["color"],
                label=series["label"],
                linewidth=2.2,
            )
        xlim = bins.get("xlim", (edges[0], edges[-1]))
        ax.set_xlim(*xlim)
        ax.set_xlabel(bins.get("xlabel", "Energy [MeV]"))
        ax.set_ylabel(bins.get("ylabel", "Tokens (arb. scale)"))
        if bins.get("log_x", True):
            ax.set_xscale("log")
        if bins.get("log_y", True):
            ax.set_yscale("log")
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            framealpha=1.0,
        )
        ax.grid(True, which="both", alpha=0.25, linestyle="--")
        ax.minorticks_off()
        fig.tight_layout()
        _save_poster_fig(fig, out_path)
        plt.close(fig)


def _plot_poster_blob_prong_energy_v2(bins: dict, out_path: Path):
    """poster_v2: blob and prong token energy spectra (log-log, MeV)."""
    primary, secondary = _poster_v2_series_colors()
    v2_colors = [primary, secondary]
    with plt.rc_context(_poster_v2_style()):
        fig, ax = plt.subplots(figsize=POSTER_ENERGY_FIGSIZE)
        edges = bins["bin_edges"]
        for idx, series in enumerate(bins["series"]):
            ax.stairs(
                _counts_to_density(bins[series["counts_key"]], edges),
                edges,
                color=v2_colors[idx % len(v2_colors)],
                label=series["label"],
                linewidth=2.2,
            )
        xlim = bins.get("xlim", (edges[0], edges[-1]))
        ax.set_xlim(*xlim)
        ax.set_xlabel(bins.get("xlabel", "Energy [MeV]"))
        ax.set_ylabel(bins.get("ylabel", "Tokens (arb. scale)"))
        if bins.get("log_x", True):
            ax.set_xscale("log")
        if bins.get("log_y", True):
            ax.set_yscale("log")
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            framealpha=1.0,
        )
        ax.grid(True, which="both", alpha=0.25, linestyle="--")
        ax.minorticks_off()
        fig.tight_layout()
        _save_poster_fig(fig, out_path, png_dpi=POSTER_V2_PNG_DPI)
        plt.close(fig)


def _plot_poster_blob_prong_multiplicity(bins: dict, out_path: Path):
    """Overlaid blob and prong multiplicity step histograms."""
    with plt.rc_context(_poster_style()):
        fig, ax = plt.subplots(figsize=POSTER_FIGSIZE)
        edges = bins["bin_edges"]
        for series in bins["series"]:
            ax.stairs(
                _counts_to_density(bins[series["counts_key"]], edges),
                edges,
                color=series["color"],
                label=series["label"],
                linewidth=2.2,
            )
        xlim = bins.get("xlim", (0.0, float(edges[-1])))
        ax.set_xlim(*xlim)
        ax.set_xlabel("Multiplicity")
        ax.set_ylabel("Events (arb. scale)")
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            framealpha=1.0,
        )
        if bins.get("log_y", True):
            ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.25, linestyle="--")
        ax.minorticks_off()
        fig.tight_layout()
        _save_poster_fig(fig, out_path)
        plt.close(fig)


def _plot_poster_blob_prong_multiplicity_v2(bins: dict, out_path: Path):
    """poster_v2: blob and prong multiplicity with two reds."""
    primary, secondary = _poster_v2_series_colors()
    v2_colors = [primary, secondary]
    with plt.rc_context(_poster_v2_style()):
        fig, ax = plt.subplots(figsize=POSTER_FIGSIZE)
        edges = bins["bin_edges"]
        for idx, series in enumerate(bins["series"]):
            ax.stairs(
                _counts_to_density(bins[series["counts_key"]], edges),
                edges,
                color=v2_colors[idx % len(v2_colors)],
                label=series["label"],
                linewidth=2.2,
            )
        xlim = bins.get("xlim", (0.0, float(edges[-1])))
        ax.set_xlim(*xlim)
        ax.set_xlabel("Multiplicity")
        ax.set_ylabel("Events (arb. scale)")
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            framealpha=1.0,
        )
        if bins.get("log_y", True):
            ax.set_yscale("log")
        ax.grid(True, which="both", axis="y", alpha=0.25, linestyle="--")
        ax.minorticks_off()
        fig.tight_layout()
        _save_poster_fig(fig, out_path, png_dpi=POSTER_V2_PNG_DPI)
        plt.close(fig)


def _plot_poster_e_available_spectrum(bins: dict, out_path: Path):
    """Step histogram of MC E_available from cached bin values."""
    with plt.rc_context(_poster_style()):
        fig, ax = plt.subplots(figsize=POSTER_ENERGY_FIGSIZE)
        edges = bins["bin_edges"]
        for series in bins["series"]:
            ax.stairs(
                _counts_to_density(bins[series["counts_key"]], edges),
                edges,
                color=series["color"],
                label=series["label"],
                linewidth=2.2,
            )
        xlim = bins.get("xlim", POSTER_ENERGY_XLIM)
        ax.set_xlim(*xlim)
        ax.set_xlabel(bins.get("xlabel", r"$E_\mathrm{available}$ [GeV]"))
        ax.set_ylabel("Events (arb. scale)")
        legend_kwargs = {
            "loc": "upper right",
            "frameon": True,
            "fancybox": False,
            "edgecolor": "0.6",
            "framealpha": 1.0,
        }
        if len(bins["series"]) > 2:
            legend_kwargs["fontsize"] = 9
        ax.legend(**legend_kwargs)
        if bins.get("log_y", False):
            ax.set_yscale("log")
            ax.grid(True, which="both", axis="y", alpha=0.25, linestyle="--")
            ax.minorticks_off()
        else:
            ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        fig.tight_layout()
        _save_poster_fig(fig, out_path)
        plt.close(fig)


def _plot_poster_e_available_spectrum_v2(bins: dict, out_path: Path):
    """poster_v2: E_available spectrum (series colors or two-red fallback)."""
    primary, secondary = _poster_v2_series_colors()
    v2_fallback = [primary, secondary]
    with plt.rc_context(_poster_v2_style()):
        fig, ax = plt.subplots(figsize=POSTER_ENERGY_FIGSIZE)
        edges = bins["bin_edges"]
        for idx, series in enumerate(bins["series"]):
            color = series.get("v2_color", v2_fallback[idx % len(v2_fallback)])
            ax.stairs(
                _counts_to_density(bins[series["counts_key"]], edges),
                edges,
                color=color,
                label=series["label"],
                linewidth=2.2,
            )
        xlim = bins.get("xlim", POSTER_ENERGY_XLIM)
        ax.set_xlim(*xlim)
        ax.set_xlabel(bins.get("xlabel", r"$E_\mathrm{available}$ [GeV]"))
        ax.set_ylabel("Events (arb. scale)")
        legend_kwargs = {
            "loc": "upper right",
            "frameon": True,
            "fancybox": False,
            "edgecolor": "0.6",
            "framealpha": 1.0,
        }
        if len(bins["series"]) > 2:
            legend_kwargs["fontsize"] = 9
        ax.legend(**legend_kwargs)
        if bins.get("log_y", False):
            ax.set_yscale("log")
            ax.grid(True, which="both", axis="y", alpha=0.25, linestyle="--")
            ax.minorticks_off()
        else:
            ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        fig.tight_layout()
        _save_poster_fig(fig, out_path, png_dpi=POSTER_V2_PNG_DPI)
        plt.close(fig)


def _e_available_spectrum_for_cache(cache: dict, args=None) -> dict | None:
    spectrum = cache.get("e_available_spectrum")
    if spectrum is not None:
        return spectrum
    if args is None:
        return None
    return _resolve_e_available_spectrum_bins(cache, args)


def _e_available_by_class_for_cache(cache: dict, args=None) -> dict | None:
    by_class = cache.get("e_available_by_class")
    if by_class is not None:
        return by_class
    if args is None:
        return None
    return _resolve_e_available_by_class_bins(cache, args)


def _blob_prong_multiplicity_for_cache(cache: dict, args=None) -> dict | None:
    if args is None:
        return cache.get("blob_prong_multiplicity")
    return _resolve_blob_prong_multiplicity_bins(cache, args)


def _blob_prong_energy_for_cache(cache: dict, args=None) -> dict | None:
    if args is None:
        return cache.get("blob_prong_energy")
    return _resolve_blob_prong_energy_bins(cache, args)


def _write_poster_plots(cache: dict, poster_dir: Path, args=None):
    tag = cache["tag"]
    cache["tokens_per_event"]["color"] = MINERVA_COLOR
    for series in cache["energy_comparison"]["series"]:
        if series["counts_key"] == "incoming_counts":
            series["label"] = "MINERvA: Neutrino energy"
            series["color"] = MINERVA_COLOR
        elif series["counts_key"] == "sum_token_counts":
            series["color"] = "#2a8a7a"
    _plot_poster_tokens_per_event(
        cache["tokens_per_event"],
        poster_dir / f"tokens_per_event_{tag}",
    )
    multiplicity = _blob_prong_multiplicity_for_cache(cache, args)
    if multiplicity is not None:
        _plot_poster_blob_prong_multiplicity(
            multiplicity,
            poster_dir / f"blob_prong_multiplicity_{tag}",
        )
    blob_prong_energy = _blob_prong_energy_for_cache(cache, args)
    if blob_prong_energy is not None:
        _plot_poster_blob_prong_energy(
            blob_prong_energy,
            poster_dir / f"blob_prong_energy_{tag}",
        )
    _plot_poster_energy_comparison(
        cache["energy_comparison"],
        poster_dir / f"energy_comparison_{tag}",
    )
    spectrum = _e_available_spectrum_for_cache(cache, args)
    if spectrum is not None:
        _plot_poster_e_available_spectrum(
            spectrum,
            poster_dir / f"e_available_spectrum_{tag}",
        )
    by_class = _e_available_by_class_for_cache(cache, args)
    if by_class is not None:
        _plot_poster_e_available_spectrum(
            by_class,
            poster_dir / f"e_available_by_class_{tag}",
        )


def _write_poster_v2_plots(cache: dict, poster_dir: Path, args=None):
    tag = cache["tag"]
    _plot_poster_tokens_per_event_v2(
        cache["tokens_per_event"],
        poster_dir / f"tokens_per_event_{tag}",
    )
    multiplicity = _blob_prong_multiplicity_for_cache(cache, args)
    if multiplicity is not None:
        _plot_poster_blob_prong_multiplicity_v2(
            multiplicity,
            poster_dir / f"blob_prong_multiplicity_{tag}",
        )
    blob_prong_energy = _blob_prong_energy_for_cache(cache, args)
    if blob_prong_energy is not None:
        _plot_poster_blob_prong_energy_v2(
            blob_prong_energy,
            poster_dir / f"blob_prong_energy_{tag}",
        )
    _plot_poster_energy_comparison_v2(
        cache["energy_comparison"],
        poster_dir / f"energy_comparison_{tag}",
    )
    spectrum = _e_available_spectrum_for_cache(cache, args)
    if spectrum is not None:
        _plot_poster_e_available_spectrum_v2(
            spectrum,
            poster_dir / f"e_available_spectrum_{tag}",
        )
    by_class = _e_available_by_class_for_cache(cache, args)
    if by_class is not None:
        _plot_poster_e_available_spectrum_v2(
            by_class,
            poster_dir / f"e_available_by_class_{tag}",
        )


def _write_jetclass2_poster_plots(jetclass2_cache: dict, poster_dir: Path):
    jc_tag = jetclass2_cache["tag"]
    jetclass2_cache["tokens_per_event"]["color"] = JETCLASS2_COLOR
    for series in jetclass2_cache["energy_comparison"]["series"]:
        if series["counts_key"] == "jet_energy_counts":
            series["color"] = JETCLASS2_COLOR
        elif series["counts_key"] == "sum_token_counts":
            series["color"] = JETCLASS2_SUM_E_COLOR
    _plot_poster_tokens_per_event(
        jetclass2_cache["tokens_per_event"],
        poster_dir / f"tokens_per_event_{jc_tag}",
    )
    _plot_poster_energy_comparison(
        jetclass2_cache["energy_comparison"],
        poster_dir / f"energy_comparison_{jc_tag}",
    )


def _write_jetclass2_poster_v2_plots(jetclass2_cache: dict, poster_dir: Path):
    jc_tag = jetclass2_cache["tag"]
    _plot_poster_tokens_per_event_v2(
        jetclass2_cache["tokens_per_event"],
        poster_dir / f"tokens_per_event_{jc_tag}",
    )
    _plot_poster_energy_comparison_v2(
        jetclass2_cache["energy_comparison"],
        poster_dir / f"energy_comparison_{jc_tag}",
    )


def _plot_joint_tokens_per_event(
    minerva_bins: dict, jetclass2_bins: dict, out_path: Path
):
    """Overlay MINERvA and JetClass-II tokens-per-event distributions."""
    with plt.rc_context(_poster_style()):
        fig, ax = plt.subplots(figsize=JOINT_POSTER_FIGSIZE)
        mv_edges = minerva_bins["bin_edges"]
        jc_edges = jetclass2_bins["bin_edges"]
        ax.stairs(
            _counts_to_density(minerva_bins["counts"], mv_edges),
            mv_edges,
            color=MINERVA_COLOR,
            linewidth=2.2,
            label="MINERvA",
        )
        ax.stairs(
            _counts_to_density(jetclass2_bins["counts"], jc_edges),
            jc_edges,
            color=JETCLASS2_COLOR,
            linewidth=2.2,
            label="JetClass2",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Tokens per event / jet")
        ax.set_ylabel("Events (arb. scale)")
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            framealpha=1.0,
        )
        ax.grid(True, which="both", axis="y", alpha=0.25, linestyle="--")
        ax.minorticks_off()
        fig.tight_layout()
        _save_poster_fig(fig, out_path)
        plt.close(fig)


def _plot_joint_tokens_per_event_v2(
    minerva_bins: dict, jetclass2_bins: dict, out_path: Path
):
    """poster_v2 joint tokens overlay with two reds."""
    primary, secondary = _poster_v2_series_colors()
    with plt.rc_context(_poster_v2_style()):
        fig, ax = plt.subplots(figsize=JOINT_POSTER_FIGSIZE)
        mv_edges = minerva_bins["bin_edges"]
        jc_edges = jetclass2_bins["bin_edges"]
        ax.stairs(
            _counts_to_density(minerva_bins["counts"], mv_edges),
            mv_edges,
            color=primary,
            linewidth=2.2,
            label="MINERvA",
        )
        ax.stairs(
            _counts_to_density(jetclass2_bins["counts"], jc_edges),
            jc_edges,
            color=secondary,
            linewidth=2.2,
            label="JetClass2",
        )
        ax.set_yscale("log")
        ax.set_xlabel("Tokens per event / jet")
        ax.set_ylabel("Events (arb. scale)")
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            framealpha=1.0,
        )
        ax.grid(True, which="both", axis="y", alpha=0.25, linestyle="--")
        ax.minorticks_off()
        fig.tight_layout()
        _save_poster_fig(fig, out_path, png_dpi=POSTER_V2_PNG_DPI)
        plt.close(fig)


def _plot_joint_reference_energy(joint_bins: dict, out_path: Path):
    """Overlay MINERvA incoming-neutrino and JetClass-II jet-energy distributions."""
    edges = joint_bins["bin_edges"]
    with plt.rc_context(_poster_style()):
        fig, ax = plt.subplots(figsize=JOINT_POSTER_FIGSIZE)
        ax.stairs(
            _counts_to_density(joint_bins["incoming_counts"], edges),
            edges,
            color=MINERVA_COLOR,
            linewidth=2.2,
            label="MINERvA: Neutrino energy",
        )
        ax.stairs(
            _counts_to_density(joint_bins["jet_energy_counts"], edges),
            edges,
            color=JETCLASS2_COLOR,
            linewidth=2.2,
            label="JetClass2: Jet energy",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        xlim = joint_bins.get("xlim", (edges[0], JOINT_ENERGY_XMAX_GEV))
        ax.set_xlim(*xlim)
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel("Events (arb. scale)")
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            framealpha=1.0,
        )
        ax.grid(True, which="both", axis="y", alpha=0.25, linestyle="--")
        ax.minorticks_off()
        fig.tight_layout()
        _save_poster_fig(fig, out_path)
        plt.close(fig)


def _plot_joint_reference_energy_v2(joint_bins: dict, out_path: Path):
    """poster_v2 joint energy overlay with two reds."""
    primary, secondary = _poster_v2_series_colors()
    edges = joint_bins["bin_edges"]
    with plt.rc_context(_poster_v2_style()):
        fig, ax = plt.subplots(figsize=JOINT_POSTER_FIGSIZE)
        ax.stairs(
            _counts_to_density(joint_bins["incoming_counts"], edges),
            edges,
            color=primary,
            linewidth=2.2,
            label="MINERvA: Neutrino energy",
        )
        ax.stairs(
            _counts_to_density(joint_bins["jet_energy_counts"], edges),
            edges,
            color=secondary,
            linewidth=2.2,
            label="JetClass2: Jet energy",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        xlim = joint_bins.get("xlim", (edges[0], JOINT_ENERGY_XMAX_GEV))
        ax.set_xlim(*xlim)
        ax.set_xlabel("Energy [GeV]")
        ax.set_ylabel("Events (arb. scale)")
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.6",
            framealpha=1.0,
        )
        ax.grid(True, which="both", axis="y", alpha=0.25, linestyle="--")
        ax.minorticks_off()
        fig.tight_layout()
        _save_poster_fig(fig, out_path, png_dpi=POSTER_V2_PNG_DPI)
        plt.close(fig)


def _write_joint_poster_plots(
    minerva_cache: dict,
    jetclass2_cache: dict,
    poster_dir: Path,
    tag: str,
    args,
):
    joint_dir = poster_dir / "joint"
    joint_dir.mkdir(parents=True, exist_ok=True)
    _plot_joint_tokens_per_event(
        minerva_cache["tokens_per_event"],
        jetclass2_cache["tokens_per_event"],
        joint_dir / f"tokens_per_event_{tag}",
    )
    incoming_gev = _resolve_incoming_e_gev(minerva_cache, args)
    jet_energy_gev = _resolve_jet_energy_gev(jetclass2_cache, args)
    if jet_energy_gev is None:
        print("Skipping joint energy plot: JetClass-II jet energy unavailable.")
        return
    joint_energy = _build_joint_reference_energy_bins(incoming_gev, jet_energy_gev)
    _plot_joint_reference_energy(
        joint_energy,
        joint_dir / f"energy_comparison_{tag}",
    )


def _write_joint_poster_v2_plots(
    minerva_cache: dict,
    jetclass2_cache: dict,
    poster_dir: Path,
    tag: str,
    args,
):
    joint_dir = poster_dir / "joint"
    joint_dir.mkdir(parents=True, exist_ok=True)
    _plot_joint_tokens_per_event_v2(
        minerva_cache["tokens_per_event"],
        jetclass2_cache["tokens_per_event"],
        joint_dir / f"tokens_per_event_{tag}",
    )
    incoming_gev = _resolve_incoming_e_gev(minerva_cache, args)
    jet_energy_gev = _resolve_jet_energy_gev(jetclass2_cache, args)
    if jet_energy_gev is None:
        print(
            "Skipping joint energy plot (poster_v2): JetClass-II jet energy unavailable."
        )
        return
    joint_energy = _build_joint_reference_energy_bins(incoming_gev, jet_energy_gev)
    _plot_joint_reference_energy_v2(
        joint_energy,
        joint_dir / f"energy_comparison_{tag}",
    )


def _plot_pid_by_type(particles: np.ndarray, out_path: Path):
    """PID counts and log_E distribution per PID."""
    pid = particles[:, 4].astype(np.int64)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    pid_names = {
        0: "muon",
        1: "photon",
        2: "blob",
        3: "prong π",
        4: "prong EM",
        5: "prong μ-like",
        6: "agg blob",
        7: "agg prong",
    }
    for p, ax in zip(range(8), axes.flat):
        mask = pid == p
        n = int(mask.sum())
        ax.set_title(f"PID {p} ({pid_names.get(p, '?')}), n={n:,}", fontsize=9)
        if n == 0:
            ax.text(
                0.5, 0.5, "no entries", ha="center", va="center", transform=ax.transAxes
            )
            ax.axis("off")
            continue
        log_e = particles[mask, 3]
        ax.hist(log_e, bins=60, color="steelblue", edgecolor="white", linewidth=0.2)
        ax.set_xlabel("log(E)", fontsize=8)
        ax.set_ylabel("count", fontsize=8)
    fig.suptitle("log(E) distribution by particle PID", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.playlist}_{args.split}"
    poster_dir = out_dir / "poster"
    poster_dir.mkdir(parents=True, exist_ok=True)
    poster_v2_dir = out_dir / "poster_v2"
    poster_v2_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _poster_cache_path(poster_dir, tag)

    if args.poster_from_cache:
        print(f"Loading poster cache {cache_path} ...")
        cache = _load_poster_cache(cache_path)
        cache.pop("jetclass2", None)
        _write_poster_plots(cache, poster_dir, args)
        _write_poster_v2_plots(cache, poster_v2_dir, args)
        jc_cache, jc_cache_path = _load_or_build_jetclass2_poster(args, poster_dir, tag)
        if jc_cache is not None:
            _write_jetclass2_poster_plots(jc_cache, poster_dir)
            _write_jetclass2_poster_v2_plots(jc_cache, poster_v2_dir)
            _write_joint_poster_plots(cache, jc_cache, poster_dir, tag, args)
            _write_joint_poster_v2_plots(cache, jc_cache, poster_v2_dir, tag, args)
        print(
            f"Wrote poster plots to {poster_dir} (joint plots in {poster_dir / 'joint'})"
        )
        print(
            f"Wrote poster_v2 plots to {poster_v2_dir} "
            f"(joint plots in {poster_v2_dir / 'joint'})"
        )
        return

    folder = Path(args.data_path) / args.playlist / args.split
    print(f"Loading {folder} ...")
    (
        particles,
        global_features,
        truth_labels,
        n_per_event,
        sum_token_energy_per_event,
    ) = _subsample_from_nested(folder, args.max_events, args.seed)
    n_events = len(n_per_event)
    n_particles = len(particles)
    print(
        f"  events={n_events:,}  particles={n_particles:,}  "
        f"particles/event: mean={n_per_event.mean():.2f} max={n_per_event.max()}"
    )

    summaries: dict[str, dict] = {
        "meta": {
            "data_path": str(folder),
            "n_events": n_events,
            "n_particles": n_particles,
            "max_events": args.max_events,
            "seed": args.seed,
        }
    }

    summaries["particles"] = _plot_grid(
        particles,
        PARTICLE_NAMES,
        PARTICLE_DISCRETE,
        out_dir / f"particle_features_{tag}.pdf",
        f"Per-particle features ({tag}, n_particles={n_particles:,})",
        args.bins,
        args.log_y,
    )
    _plot_pid_by_type(particles, out_dir / f"logE_by_PID_{tag}.pdf")

    summaries["global_features"] = _plot_grid(
        global_features,
        GLOBAL_NAMES,
        GLOBAL_DISCRETE,
        out_dir / f"global_features_{tag}.pdf",
        f"Global / conditioning features ({tag}, n_events={n_events:,})",
        args.bins,
        args.log_y,
    )

    summaries["truth_labels"] = _plot_grid(
        truth_labels,
        TRUTH_NAMES,
        TRUTH_DISCRETE,
        out_dir / f"truth_labels_{tag}.pdf",
        f"Truth labels ({tag}, n_events={n_events:,})",
        args.bins,
        args.log_y,
        discrete_xlabels={1: INT_TYPE_LABELS, 3: CURRENT_LABELS},
        sentinels=TRUTH_SENTINEL,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        n_per_event,
        bins=np.arange(0, max(n_per_event.max() + 2, 2)),
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("particles per event")
    ax.set_ylabel("count")
    ax.set_title(f"Event multiplicity ({tag}, n_events={n_events:,})")
    if args.log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / f"n_particles_per_event_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    summaries["n_particles_per_event"] = _summary(n_per_event.astype(np.float64))

    jetclass2_payload = None
    jc_cache_path = None
    jc_cache = None
    if not args.no_jetclass2:
        jc_cache, jc_cache_path = _load_or_build_jetclass2_poster(args, poster_dir, tag)
        if jc_cache is not None:
            jetclass2_payload = jc_cache["meta"]
            _write_jetclass2_poster_plots(jc_cache, poster_dir)
            _write_jetclass2_poster_v2_plots(jc_cache, poster_v2_dir)

    blob_counts, prong_counts = _blob_prong_counts_per_event(particles, n_per_event)
    blob_e, prong_e = _blob_prong_token_energies_mev(particles)
    poster_cache = _build_poster_cache(
        n_per_event,
        truth_labels[:, 0],
        sum_token_energy_per_event,
        truth_labels[:, E_AVAILABLE_WITH_MUON_COL],
        truth_labels[:, E_AVAILABLE_NO_MUON_COL],
        truth_labels,
        blob_counts,
        prong_counts,
        blob_e,
        prong_e,
        tag,
        meta={
            "data_path": str(folder),
            "playlist": args.playlist,
            "split": args.split,
            "n_events": n_events,
            "max_events": args.max_events,
            "seed": args.seed,
        },
    )
    _save_poster_cache(poster_cache, cache_path)
    _write_poster_plots(poster_cache, poster_dir, args)
    _write_poster_v2_plots(poster_cache, poster_v2_dir, args)
    if not args.no_jetclass2 and jc_cache is not None:
        _write_joint_poster_plots(poster_cache, jc_cache, poster_dir, tag, args)
        _write_joint_poster_v2_plots(poster_cache, jc_cache, poster_v2_dir, tag, args)
    summaries["poster"] = {
        "cache_path": str(cache_path),
        "tokens_per_event": _summary(n_per_event.astype(np.float64)),
        "mc_incomingE_MeV": _summary(truth_labels[:, 0]),
        "sum_token_energy_MeV": _summary(sum_token_energy_per_event),
        "E_available_with_muon_MeV": _summary(
            truth_labels[:, E_AVAILABLE_WITH_MUON_COL]
        ),
        "E_available_no_muon_MeV": _summary(truth_labels[:, E_AVAILABLE_NO_MUON_COL]),
    }
    if jetclass2_payload is not None and jc_cache_path is not None:
        summaries["poster"]["jetclass2"] = {
            "cache_path": str(jc_cache_path),
            "files": jetclass2_payload["files"],
            "n_events": jetclass2_payload["n_events"],
        }
    print(f"Wrote poster plots and cache to {poster_dir}")
    print(f"Wrote poster_v2 plots to {poster_v2_dir}")
    if not args.no_jetclass2 and jc_cache is not None:
        print(f"  joint plots in {poster_dir / 'joint'} and {poster_v2_dir / 'joint'}")

    summary_path = out_dir / f"summary_{tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Wrote plots and {summary_path}")


if __name__ == "__main__":
    main()
