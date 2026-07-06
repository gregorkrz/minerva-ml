#!/usr/bin/env python3
"""Extract per-bin signal / background sample "datasets" for a lightweight demo.

For a chosen playlist test split this script slices the ML-ready dataset (see
``DATASET.md``) into many small ``.pb`` datasets, one per *(classification task,
kinematic bin, signal|background)* combination, keeping the **exact same file
format** as the main dataset (a dict with ``data`` jagged nested tensor,
``truth_labels`` and ``global_features`` tensors). These are intended to seed a
live demo that shows event displays and per-model output scores.

Three classification tasks are produced, matching the eval plotting scripts
(``src/eval/plot_classification_W.py`` and ``plot_classification_Pions.py``):

* ``ccnpi_W``      — CCN\u03c0\u00b1 (N\u22651) tagging, binned in true hadronic *W* [GeV].
                     Signal = ``pid \u2208 {0, 1}`` (CC 1 charged \u03c0 or CC >1 charged \u03c0),
                     background = ``pid \u2208 {2, 3, 4}``.
* ``cc1pi_pionE``  — CC1\u03c0\u00b1 tagging, binned in true pion energy [GeV].
                     Signal = ``pid == 0``, background = everything else.
* ``cc1pi0_pionE`` — CC1\u03c0\u2070 tagging, binned in true pion energy [GeV].
                     Signal = ``pid == 2``, background = everything else.

Bin definitions reuse the same code paths as the plots:

* *W* bins: fixed ``DEFAULT_W_BIN_EDGES_GEV`` (= ``[0, .5, 1, ..., 4]``), with
  per-event MC truth *W* from ``mc_true_hadronic_W_GeV`` in the baselines file.
* Pion-energy bins: equal-frequency edges built from the **true signal** pion
  energies only (same as ``data_with_signal_pion_bins(... method="equal_frequency")``).
  Background events are assigned to those same edges by their MC pion energy
  (``pion_four_vectors`` is nonzero only for single-pion CC events, so most
  non-signal events fall outside the bins and contribute no background — this
  mirrors the plots' ``pion_bins_require_has_pion=False`` behaviour).

Empty (signal- or background-less) bins are skipped, as requested.

Output layout (``--output-dir``)::

    <out>/manifest.json
    <out>/ccnpi_W/bin00_0.000-0.500GeV/signal/0.pb
    <out>/ccnpi_W/bin00_0.000-0.500GeV/signal/meta.json
    <out>/ccnpi_W/bin00_0.000-0.500GeV/background/0.pb
    ...

Each ``<...>/0.pb`` is directly loadable with
``src.dataset.dataloader.HEPTorchDataset(folder=<that directory>)``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.classification_plots._binning import equal_frequency_bin_edges
from src.eval.classification_plots._constants import DEFAULT_W_BIN_EDGES_GEV
from src.eval.classification_plots._hadronic_w import (
    mc_true_hadronic_W_gev_from_baselines,
)

DEFAULT_DATA_DIR = Path("/global/cfs/cdirs/m3246/gregork/Minerva/20260326_NEW")
DEFAULT_N_PION_BINS = 5

# Pi_labels_v2 signal classes (see src/eval/classification_plots/_signal_definitions.py).
CCNPI_GE1_CLASSES = [0, 1]
CC1PI_CLASSES = [0]
CC1PI0_CLASSES = [2]


def get_pi_labels_v2(truth_labels: np.ndarray) -> np.ndarray:
    """Pi_labels_v2 per event (delegates to ``dataloader.get_Pi_labels_v2``)."""
    from src.dataset.dataloader import get_Pi_labels_v2

    tl = np.asarray(truth_labels)
    return get_Pi_labels_v2(torch.from_numpy(tl)).numpy()


def value_in_bin(x: np.ndarray, edges: np.ndarray, bin_index: int) -> np.ndarray:
    """Histogram-bin membership matching ``_metrics_binned.mc_value_in_bin``.

    Interior bins are half-open ``[lo, hi)``; the last bin is closed ``[lo, hi]``.
    NaN values never match (comparisons evaluate False).
    """
    edges = np.asarray(edges, dtype=float)
    n_bins = len(edges) - 1
    lo, hi = float(edges[bin_index]), float(edges[bin_index + 1])
    if bin_index == n_bins - 1:
        return (x >= lo) & (x <= hi)
    return (x >= lo) & (x < hi)


def load_split(
    data_dir: Path, playlist: str, split: str
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, np.ndarray, dict]:
    """Load the ``.pb`` split plus aligned ``test_idx`` and baselines.

    Returns ``(per_event_data, truth_labels, global_features, split_idx, baselines)``
    where ``per_event_data`` is a list of ``(n_particles, 10)`` tensors.
    """
    pb_path = data_dir / playlist / split / "0.pb"
    if not pb_path.exists():
        raise FileNotFoundError(f"Missing split file: {pb_path}")
    blob = torch.load(pb_path, weights_only=False, map_location="cpu")
    per_event = list(blob["data"].unbind())
    truth = blob["truth_labels"]
    if not torch.is_tensor(truth):
        truth = torch.as_tensor(truth)
    glob = blob["global_features"]
    if not torch.is_tensor(glob):
        glob = torch.as_tensor(glob)

    with open(data_dir / "result.pkl", "rb") as f:
        result = pickle.load(f)
    if playlist not in result:
        raise KeyError(f"Playlist {playlist!r} not in {data_dir/'result.pkl'}")
    split_idx = np.asarray(result[playlist][f"{split}_idx"])

    bl_path = data_dir / "baselines" / f"{playlist}_enu_baselines.npz"
    if not bl_path.exists():
        raise FileNotFoundError(f"Missing baselines file: {bl_path}")
    baselines = dict(np.load(bl_path))

    n = len(per_event)
    if not (len(truth) == len(glob) == n == len(split_idx)):
        raise ValueError(
            f"Length mismatch: data={n}, truth={len(truth)}, "
            f"global={len(glob)}, {split}_idx={len(split_idx)}. "
            "The split .pb and result.pkl are not aligned."
        )
    return per_event, truth, glob, split_idx, baselines


def save_subset(
    per_event: list[torch.Tensor],
    truth: torch.Tensor,
    glob: torch.Tensor,
    idx: np.ndarray,
    out_dir: Path,
) -> None:
    """Write a subset as ``out_dir/0.pb`` in the canonical dataset format."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_list = [int(i) for i in idx]
    # Clone selected per-event tensors so the saved file does not pull in the
    # entire backing storage of the original jagged tensor.
    selected = [per_event[i].clone() for i in idx_list]
    subset = {
        "data": torch.nested.nested_tensor(selected, layout=torch.jagged),
        "truth_labels": truth[idx_list].clone(),
        "global_features": glob[idx_list].clone(),
    }
    torch.save(subset, out_dir / "0.pb")


def _bin_label(i: int, lo: float, hi: float, unit: str) -> str:
    return f"bin{i:02d}_{lo:.3f}-{hi:.3f}{unit}"


def process_task(
    *,
    task_name: str,
    title: str,
    signal_classes: list[int],
    bin_var: np.ndarray,
    bin_edges: np.ndarray,
    bin_unit: str,
    pid: np.ndarray,
    per_event: list[torch.Tensor],
    truth: torch.Tensor,
    glob: torch.Tensor,
    split_idx: np.ndarray,
    out_root: Path,
    n_events: int,
    rng: np.random.Generator,
) -> dict:
    """Slice and save signal/background samples for one task, return its manifest."""
    signal_set = set(signal_classes)
    is_signal = np.isin(pid, list(signal_set))
    n_bins = len(bin_edges) - 1
    task_dir = out_root / task_name

    task_entry: dict = {
        "title": title,
        "signal_pid_classes": signal_classes,
        "bin_variable": bin_unit.strip() or "value",
        "bin_edges": [float(e) for e in bin_edges],
        "bins": [],
    }
    print(f"\n=== {task_name}: {title} ===")
    print(f"    bin edges ({bin_unit.strip()}): "
          + ", ".join(f"{e:.3f}" for e in bin_edges))

    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        in_bin = value_in_bin(bin_var, bin_edges, i)
        label = _bin_label(i, lo, hi, bin_unit)

        bin_entry: dict = {"index": i, "lo": lo, "hi": hi, "label": label}
        any_written = False
        for cls_name, cls_mask in (
            ("signal", in_bin & is_signal),
            ("background", in_bin & ~is_signal),
        ):
            pool = np.flatnonzero(cls_mask)
            if pool.size == 0:
                print(f"    [{label}] {cls_name}: 0 available -> skip")
                continue
            chosen = pool.copy()
            rng.shuffle(chosen)
            chosen = chosen[:n_events]
            chosen = np.sort(chosen)
            out_dir = task_dir / label / cls_name
            save_subset(per_event, truth, glob, chosen, out_dir)

            pid_sel = pid[chosen]
            meta = {
                "task": task_name,
                "class": cls_name,
                "bin_index": i,
                "bin_lo": lo,
                "bin_hi": hi,
                "bin_unit": bin_unit.strip(),
                "n_events": int(chosen.size),
                "n_available": int(pool.size),
                "pid_classes": [int(p) for p in pid_sel],
                "pid_counts": {
                    int(c): int((pid_sel == c).sum()) for c in np.unique(pid_sel)
                },
                "bin_values": [float(v) for v in bin_var[chosen]],
                # Index into the playlist test split (0..N_test-1) and into the
                # full per-playlist arrays (baselines / ROOT order).
                "split_local_index": [int(c) for c in chosen],
                "playlist_global_index": [int(split_idx[c]) for c in chosen],
            }
            with open(out_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            print(
                f"    [{label}] {cls_name}: wrote {chosen.size} "
                f"(of {pool.size} available)"
            )
            bin_entry[cls_name] = {
                "path": str(out_dir.relative_to(out_root)),
                "n_events": int(chosen.size),
                "n_available": int(pool.size),
            }
            any_written = True

        if any_written:
            task_entry["bins"].append(bin_entry)
    return task_entry


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help=f"Dataset root (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--playlist", default="1A", help="Playlist (default: 1A)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"],
                    help="Which split to sample from (default: test)")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write the per-bin sample datasets")
    ap.add_argument("--n-events", type=int, default=10,
                    help="Max events per (bin, signal|background) (default: 10)")
    ap.add_argument("--n-pion-bins", type=int, default=DEFAULT_N_PION_BINS,
                    help=f"Pion-energy bins (default: {DEFAULT_N_PION_BINS})")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    out_root = args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.split} split for playlist {args.playlist} from {args.data_dir}")
    per_event, truth, glob, split_idx, baselines = load_split(
        args.data_dir, args.playlist, args.split
    )
    truth_np = truth.numpy()
    n_events_total = len(per_event)
    print(f"Loaded {n_events_total} events.")

    pid = get_pi_labels_v2(truth_np)
    W_GeV = mc_true_hadronic_W_gev_from_baselines(baselines, split_idx)
    pion_fv = baselines["pion_four_vectors"][split_idx] / 1000.0
    pion_E = pion_fv[:, -1]

    # Diagnostic: overall pid composition.
    pid_unique, pid_counts = np.unique(pid, return_counts=True)
    print("pid composition: "
          + ", ".join(f"{int(c)}:{int(n)}" for c, n in zip(pid_unique, pid_counts)))

    manifest: dict = {
        "data_dir": str(args.data_dir),
        "playlist": args.playlist,
        "split": args.split,
        "n_events_in_split": int(n_events_total),
        "n_events_per_bin_class": int(args.n_events),
        "seed": int(args.seed),
        "pid_label_meaning": {
            0: "CC 1 charged pion",
            1: "CC >1 charged pion",
            2: "CC 1 pi0, no charged pions",
            3: "CC other",
            4: "NC",
        },
        "format_note": (
            "Each <task>/<bin>/<signal|background>/0.pb is a dict with keys "
            "data (jagged nested tensor, N x variable x 10), truth_labels "
            "(N x 15) and global_features (N x 16); same format as DATASET.md. "
            "meta.json next to each 0.pb lists per-event pid, bin value and "
            "original split/global indices."
        ),
        "tasks": {},
    }

    # --- Task 1: CCNpi (N>=1) in W bins ---
    manifest["tasks"]["ccnpi_W"] = process_task(
        task_name="ccnpi_W",
        title="CCNpi+- (N>=1) tagging vs true hadronic W",
        signal_classes=CCNPI_GE1_CLASSES,
        bin_var=W_GeV,
        bin_edges=DEFAULT_W_BIN_EDGES_GEV.copy(),
        bin_unit="GeV",
        pid=pid,
        per_event=per_event,
        truth=truth,
        glob=glob,
        split_idx=split_idx,
        out_root=out_root,
        n_events=args.n_events,
        rng=rng,
    )

    # --- Tasks 2 & 3: CC1pi+- and CC1pi0 in pion-energy bins ---
    for task_name, title, classes in (
        ("cc1pi_pionE", "CC1pi+- tagging vs true pion energy", CC1PI_CLASSES),
        ("cc1pi0_pionE", "CC1pi0 tagging vs true pion energy", CC1PI0_CLASSES),
    ):
        sig_mask = np.isin(pid, classes) & np.isfinite(pion_E)
        n_sig_finite = int(sig_mask.sum())
        if n_sig_finite < args.n_pion_bins:
            print(
                f"\n=== {task_name}: SKIPPED (only {n_sig_finite} finite-E "
                f"signal events, need >= {args.n_pion_bins}) ==="
            )
            continue
        edges = equal_frequency_bin_edges(pion_E, sig_mask, args.n_pion_bins)
        manifest["tasks"][task_name] = process_task(
            task_name=task_name,
            title=title,
            signal_classes=classes,
            bin_var=pion_E,
            bin_edges=edges,
            bin_unit="GeV",
            pid=pid,
            per_event=per_event,
            truth=truth,
            glob=glob,
            split_idx=split_idx,
            out_root=out_root,
            n_events=args.n_events,
            rng=rng,
        )

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest: {out_root / 'manifest.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
