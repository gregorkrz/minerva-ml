"""Load checkpoints / baselines for classification evaluation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from ._constants import DEFAULT_N_BINS, DEFAULT_Q3_BIN_EDGES
from ._binning import _as_strictly_increasing_bin_edges


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_results(
    ckpt_dir: str | Path,
    training_names: dict[str, list[str]],
    playlists: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, list[dict[str, dict]]]:
    """Load ``.npz`` evaluation results for every model and run.

    Parameters
    ----------
    ckpt_dir : root checkpoint directory.
    training_names : ``{model_name: [run_name_1, run_name_2, ...]}``.
    playlists : evaluation playlists (default ``["1A"]``).

    Returns
    -------
    ``{model_name: [run0_results, run1_results, ...]}`` where each
    ``runN_results`` is ``{playlist: {prediction, pid, ...}}``.
    """
    ckpt_dir = Path(ckpt_dir)
    if playlists is None:
        playlists = ["1A"]

    all_results: dict[str, list[dict]] = {}
    for model_name in sorted(training_names.keys()):
        run_names = training_names[model_name]
        all_results[model_name] = []
        for run_name in run_names:
            run_results: dict[str, dict] = {}
            for playlist in playlists:
                p1 = (
                    ckpt_dir
                    / run_name
                    / "test_results"
                    / f"outputs_{run_name}_minerva_{playlist}_0.npz"
                )
                p2 = (
                    ckpt_dir
                    / run_name
                    / "test_results"
                    / f"outputs_best_model_minerva_{playlist}_0.npz"
                )
                if p1.exists():
                    run_results[playlist] = dict(np.load(p1))
                elif p2.exists():
                    run_results[playlist] = dict(np.load(p2))
                else:
                    raise FileNotFoundError(
                        f"No results for {run_name} playlist {playlist}: "
                        f"tried {p1} and {p2}"
                    )
                run_results[playlist]["prediction"] = _softmax(
                    run_results[playlist]["prediction"]
                )
            all_results[model_name].append(run_results)
        if verbose:
            print(f"[{model_name}] loaded {len(run_names)} run(s)")
    return all_results


def load_truth_and_baselines(
    ckpt_dir: str | Path,
    training_names: dict[str, list[str]],
    playlists: list[str] | None = None,
    n_pion_bins: int = DEFAULT_N_BINS,
    q3_bin_edges: np.ndarray | None = None,
    *,
    pion_E_bin_edges: np.ndarray | Sequence[float] | None = None,
    pion_theta_bin_edges: np.ndarray | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Load truth labels, baselines and derived kinematic arrays.

    Uses the first run of the first model to locate the data path.

    **Pion kinematic bins** default to ``np.quantile`` on events with
    ``has_pion`` (and :math:`E>0` / finite :math:`\\theta`), with *n_pion_bins*
    bins. Alternatively pass both *pion_E_bin_edges* and *pion_theta_bin_edges*
    (strictly increasing, length :math:`N+1` each) to use fixed edges; *n_pion_bins*
    is then ignored for pion binning. Same validation as
    :func:`data_with_signal_pion_bins` ``custom`` mode.

    Returns
    -------
    dict with keys: ``truth_labels``, ``test_idx``, ``baselines``,
    ``pion_E_MC``, ``pion_theta_MC``, ``has_pion``, ``q3_GeV``,
    ``int_type_arr``, ``pion_E_MC_bins``, ``pion_E_MC_bins_mid``,
    ``pion_theta_MC_bins``, ``pion_theta_MC_bins_mid``,
    ``q3_bin_edges``, ``q3_bin_mids``.
    """
    ckpt_dir = Path(ckpt_dir)
    if playlists is None:
        playlists = ["1A"]
    if q3_bin_edges is None:
        q3_bin_edges = DEFAULT_Q3_BIN_EDGES.copy()

    first_model = next(iter(training_names))
    first_run = training_names[first_model][0]

    # Find data_path from best_model.pt or settings.json
    data_path = None
    model_pt = ckpt_dir / first_run / "best_model.pt"
    if model_pt.exists():
        ckpt = torch.load(model_pt, weights_only=False, map_location="cpu")
        data_path = Path(ckpt["args"]["data_path"])
    else:
        settings_path = ckpt_dir / first_run / "settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                cfg = json.load(f)
            data_path = Path(cfg.get("data_path", cfg.get("dataset_path", "")))
    if data_path is None:
        raise FileNotFoundError(
            f"Cannot determine data_path from {first_run}: "
            "no best_model.pt or settings.json found"
        )

    split_idx = pickle.load(open(data_path / "result.pkl", "rb"))

    truth_labels: dict[str, torch.Tensor] = {}
    baselines_dict: dict[str, dict] = {}
    test_idx_dict: dict[str, np.ndarray] = {}

    for playlist in playlists:
        test_idx = split_idx[playlist]["test_idx"]
        test_idx_dict[playlist] = test_idx

        truth_labels[playlist] = torch.load(
            open(data_path / playlist / "test" / "0.pb", "rb"),
            weights_only=False,
            map_location="cpu",
        )["truth_labels"]

        baseline_file = f"{playlist}_enu_baselines.npz"
        loaded = False
        subdir = "baselines"
        candidate = data_path / subdir / baseline_file
        print("Baseline file: ", candidate)
        if candidate.exists():
            baselines_dict[playlist] = dict(np.load(candidate))
            loaded = True
            break
        if not loaded:
            print(f"[{playlist}] WARNING: no baselines found in {data_path}")

    playlist = playlists[0]
    test_idx = test_idx_dict[playlist]
    tl = truth_labels[playlist]

    # Pion kinematics
    pion_fv = baselines_dict[playlist]["pion_four_vectors"][test_idx] / 1000.0
    pion_E_MC = pion_fv[:, -1]
    has_pion = pion_fv[:, -1] > 0
    pion_p_MC = np.linalg.norm(pion_fv[:, 1:4], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pion_theta_MC = np.arccos(pion_fv[:, 2] / pion_p_MC)

    if (pion_E_bin_edges is None) ^ (pion_theta_bin_edges is None):
        raise ValueError(
            "pion_E_bin_edges and pion_theta_bin_edges must both be set or both be None"
        )
    if pion_E_bin_edges is not None:
        pion_E_bins = _as_strictly_increasing_bin_edges(
            pion_E_bin_edges, "pion_E_bin_edges"
        )
        pion_theta_bins = _as_strictly_increasing_bin_edges(
            pion_theta_bin_edges, "pion_theta_bin_edges"
        )
    else:
        # Pion bins (quantile-based on events with a pion)
        pion_E_signal = pion_E_MC[has_pion & (pion_E_MC > 0)]
        pion_theta_signal = pion_theta_MC[has_pion & np.isfinite(pion_theta_MC)]
        pion_E_bins = np.quantile(pion_E_signal, np.linspace(0, 1, n_pion_bins + 1))
        pion_theta_bins = np.quantile(
            pion_theta_signal, np.linspace(0, 1, n_pion_bins + 1)
        )

    # q3
    q3_GeV = baselines_dict[playlist]["q3"][test_idx] / 1000.0

    # Interaction type
    int_type_arr = tl[:, 1].numpy()

    return {
        "truth_labels": truth_labels,
        "test_idx": test_idx_dict,
        "baselines": baselines_dict,
        "pion_E_MC": pion_E_MC,
        "pion_theta_MC": pion_theta_MC,
        "has_pion": has_pion,
        "q3_GeV": q3_GeV,
        "int_type_arr": int_type_arr,
        "pion_E_MC_bins": pion_E_bins,
        "pion_E_MC_bins_mid": (pion_E_bins[:-1] + pion_E_bins[1:]) / 2,
        "pion_theta_MC_bins": pion_theta_bins,
        "pion_theta_MC_bins_mid": (pion_theta_bins[:-1] + pion_theta_bins[1:]) / 2,
        "q3_bin_edges": q3_bin_edges,
        "q3_bin_mids": (q3_bin_edges[:-1] + q3_bin_edges[1:]) / 2,
    }
