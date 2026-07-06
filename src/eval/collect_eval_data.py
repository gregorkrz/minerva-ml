#!/usr/bin/env python3
"""Load classification and/or regression eval inputs and cache them as pickles.

Default layout (relative to repo root)::

    out/classification_<flag>.pkl
    out/regression_<flag>.pkl

Each file is a versioned dict suitable for offline ``plot_*.py`` scripts.

**Regression pickle** (``schema_version`` 2+) also stores precomputed
``eval_data_*`` dicts returned by :func:`src.eval.e_available_plots.load_eval_data`
/ ``load_eval_data_grouped`` (``results``, ``E_true_dict``, ``E_pred_dict``,
``Enu_baselines``, ``Enu_filters``, ``mc_E``, …) so ``plot_regression.py`` can
pass ``data=...`` and avoid re-reading NPZs from ``ckpt_dir``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval._constants import (
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_CKPT_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_WANDB_TAG,
    CLRS_CLASSIFICATION,
    CLRS_REGRESSION,
    FLOPS_PER_STEP,
    REGRESSION_PICKLE_STEM,
)
from src.eval.baseline_val_loss import RECO_BASELINE_MODEL_KEY
from src.eval._wandb_loss import (
    classification_runs_per_model,
    collect_histories_for_runs,
    regression_runs_per_model,
)

SCHEMA_VERSION_CLASSIFICATION = 1
# Regression gained cached eval arrays in schema 2.
SCHEMA_VERSION_REGRESSION = 2


def _cap_to_label(cap: int) -> str:
    if cap == -1:
        return "6M"
    if cap >= 1_000_000:
        return f"{cap // 1_000_000}M"
    return f"{cap // 1000}k"


def _default_paths(out_dir: Path, flag: str) -> tuple[Path, Path]:
    return (
        out_dir / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl",
        out_dir / f"{REGRESSION_PICKLE_STEM}_{flag}.pkl",
    )


def _load_existing_loss_histories(pickle_path: Path | None) -> dict:
    """Return ``loss_histories`` from an existing eval pickle, or {}."""
    if pickle_path is None or not pickle_path.is_file():
        return {}
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return dict(data.get("loss_histories") or {})


def _strip_reco_baseline(loss_histories: dict) -> None:
    loss_histories.pop(RECO_BASELINE_MODEL_KEY, None)


def _filter_runs_by_eval_npz(
    runs_by_model_cap: dict[str, dict[int, list[str]]],
    ckpt_dir: Path,
    playlists: list[str],
) -> tuple[dict[str, dict[int, list[str]]], list[str]]:
    """Drop wandb runs that do not yet have test_results npz for every playlist."""
    from src.eval.classification_plots._io import run_has_test_results_npz

    skipped: list[str] = []
    filtered: dict[str, dict[int, list[str]]] = {}
    for model, caps in runs_by_model_cap.items():
        kept_caps: dict[int, list[str]] = {}
        for cap, run_list in caps.items():
            kept_runs = []
            for run_name in run_list:
                if run_has_test_results_npz(ckpt_dir, run_name, playlists):
                    kept_runs.append(run_name)
                else:
                    skipped.append(run_name)
            if kept_runs:
                kept_caps[cap] = kept_runs
        if kept_caps:
            filtered[model] = kept_caps
    return filtered, skipped


DEFAULT_DATA_PATH = "/global/cfs/cdirs/m3246/gregork/Minerva/20260326_NEW"


def collect_classification(
    ckpt_dir: Path,
    wandb_tag: str,
    *,
    data_path: Path | None = None,
    skip_wandb: bool = False,
    skip_unevaluated: bool = False,
    existing_pickle: Path | None = None,
) -> dict:
    from src.eval.classification_plots import (
        add_hadronic_W_to_classification_data,
        load_results,
        load_truth_and_baselines,
    )
    from src.utils.utils import get_classification_runs_by_model_and_cap

    runs_by_model_cap = get_classification_runs_by_model_and_cap(wandb_tag)
    print(
        "  Models from wandb (classification):",
        ", ".join(sorted(runs_by_model_cap.keys())) or "(none)",
    )
    playlists = ["1A", "1B"]
    if skip_unevaluated:
        runs_by_model_cap, skipped = _filter_runs_by_eval_npz(
            runs_by_model_cap, ckpt_dir, playlists
        )
        if skipped:
            print(
                "  Skipping unevaluated classification run(s) (--skip-unevaluated):",
                ", ".join(sorted(skipped)),
            )
        print(
            "  Models with eval npz (classification):",
            ", ".join(sorted(runs_by_model_cap.keys())) or "(none)",
        )
    training_names = {
        key: value[-1]
        for key, value in sorted(runs_by_model_cap.items(), key=lambda kv: kv[0])
    }
    if not training_names:
        raise SystemExit(
            "No evaluated classification runs found. "
            "Run evaluate_single_gpu first or pass --skip-unevaluated only when "
            "some runs already have test_results npz."
        )
    results = load_results(ckpt_dir, training_names, playlists=playlists)
    data_by_playlist = {
        pl: load_truth_and_baselines(ckpt_dir, training_names, playlists=[pl])
        for pl in playlists
    }
    data_w_by_playlist = {
        pl: add_hadronic_W_to_classification_data(data_by_playlist[pl], pl)
        for pl in playlists
    }

    runs_per_model = classification_runs_per_model(runs_by_model_cap, training_names)
    if skip_wandb:
        loss_histories = _load_existing_loss_histories(existing_pickle)
        print(
            "  Skipping wandb loss-history download (--skip-wandb); "
            f"kept {len(loss_histories)} model(s) from existing pickle."
        )
    else:
        loss_histories = collect_histories_for_runs(runs_per_model)
        print(
            "  Loss histories (classification, non-empty eval_loss):",
            ", ".join(sorted(loss_histories.keys())) or "(none)",
        )

    _strip_reco_baseline(loss_histories)

    return {
        "schema_version": SCHEMA_VERSION_CLASSIFICATION,
        "wandb_tag": wandb_tag,
        "ckpt_dir": ckpt_dir,
        "runs_by_model_cap": runs_by_model_cap,
        "training_names": training_names,
        "playlists": playlists,
        "results": results,
        "data_by_playlist": data_by_playlist,
        "data_w_by_playlist": data_w_by_playlist,
        "flops_per_step": dict(FLOPS_PER_STEP),
        "clrs_dict_full": dict(CLRS_CLASSIFICATION),
        "loss_histories": loss_histories,
        "use_global_fpr_W": True,
    }


def collect_regression(
    ckpt_dir: Path,
    wandb_tag: str,
    suppress_errors: bool,
    *,
    data_path: Path | None = None,
    skip_wandb: bool = False,
    skip_unevaluated: bool = False,
    existing_pickle: Path | None = None,
) -> dict:
    from src.eval.e_available_plots import load_eval_data, load_eval_data_grouped
    from src.utils.utils import get_runs_by_model_and_cap

    runs_by_model_cap = get_runs_by_model_and_cap(wandb_tag)
    print(
        "  Models from wandb (regression):",
        ", ".join(sorted(runs_by_model_cap.keys())) or "(none)",
    )
    if skip_unevaluated:
        runs_by_model_cap, skipped = _filter_runs_by_eval_npz(
            runs_by_model_cap, ckpt_dir, ["1A", "1B"]
        )
        if skipped:
            print(
                "  Skipping unevaluated regression run(s) (--skip-unevaluated):",
                ", ".join(sorted(skipped)),
            )
        print(
            "  Models with eval npz (regression):",
            ", ".join(sorted(runs_by_model_cap.keys())) or "(none)",
        )
        suppress_errors = True
    training_names_full: dict = {"Log1p": {}}
    baseline_run = None
    for model, caps in runs_by_model_cap.items():
        if -1 in caps.keys():
            run = caps[-1]
            training_names_full["Log1p"][model] = run
            if baseline_run is None:
                baseline_run = run[-1]

    if not training_names_full["Log1p"]:
        raise SystemExit("No runs with cap=-1 found. Check wandb tag.")
    if baseline_run is None:
        baseline_run = list(training_names_full["Log1p"].values())[0]

    # Historically this dict excluded any model name containing "rw" (intended to drop
    # random-weights ablations). That also removed OmniLearned-small-rw, which is a
    # first-class model whose key ends in "-rw". Include every cap=-1 model so
    # IQR/MPV and related plots match the full regression lineup.
    training_names_full_no_rw = {"Log1p": dict(training_names_full["Log1p"])}

    training_names_full_first_only: dict = {}
    for loss, models in training_names_full.items():
        training_names_full_first_only[loss] = {}
        for model, runs in models.items():
            if isinstance(runs, list):
                if not runs:
                    continue
                training_names_full_first_only[loss][model] = sorted(runs)[0]
            else:
                training_names_full_first_only[loss][model] = runs

    runs_per_model = regression_runs_per_model(runs_by_model_cap, training_names_full)
    if skip_wandb:
        loss_histories = _load_existing_loss_histories(existing_pickle)
        print(
            "  Skipping wandb loss-history download (--skip-wandb); "
            f"kept {len(loss_histories)} model(s) from existing pickle."
        )
    else:
        loss_histories = collect_histories_for_runs(runs_per_model)
        print(
            "  Loss histories (regression, non-empty eval_loss):",
            ", ".join(sorted(loss_histories.keys())) or "(none)",
        )

    _strip_reco_baseline(loss_histories)

    # Grouped layout for multi-seed models (matches plot_rms_iqr_with_uncertainty).
    grouped_no_rw: dict[str, dict[str, list[str]]] = {"Log1p": {}}
    for model, runs in training_names_full_no_rw["Log1p"].items():
        grouped_no_rw["Log1p"][model] = runs if isinstance(runs, list) else [runs]

    print("Loading regression eval arrays (IQR bundle, playlists 1A+1B)…")
    eval_data_no_rw = load_eval_data_grouped(
        ckpt_dir,
        grouped_no_rw,
        playlists=["1A", "1B"],
        baseline_run=baseline_run,
        verbose=False,
        suppress_errors=suppress_errors,
    )

    print("Loading regression eval arrays (first seed per model, 1A)…")
    eval_data_first_only = load_eval_data(
        ckpt_dir,
        training_names_full_first_only,
        playlists=["1A"],
        baseline_run=baseline_run,
        verbose=False,
        suppress_errors=suppress_errors,
    )

    training_names_grouped = {"Log1p": {}}
    for model, caps in runs_by_model_cap.items():
        for cap, run_list in sorted(
            caps.items(), key=lambda x: (x[0] == -1, -x[0] if x[0] > 0 else 0)
        ):
            if not run_list:
                continue
            label = f"{model} {_cap_to_label(cap)}"
            training_names_grouped["Log1p"][label] = run_list

    if training_names_grouped["Log1p"]:
        print("Loading regression eval arrays (dataset-size scaling, 1A)…")
        eval_data_scaling = load_eval_data_grouped(
            ckpt_dir,
            training_names_grouped,
            playlists=["1A"],
            baseline_run=baseline_run,
            verbose=False,
            suppress_errors=suppress_errors,
        )
    else:
        eval_data_scaling = None

    return {
        "schema_version": SCHEMA_VERSION_REGRESSION,
        "wandb_tag": wandb_tag,
        "ckpt_dir": ckpt_dir,
        "suppress_errors": suppress_errors,
        "runs_by_model_cap": runs_by_model_cap,
        "training_names_full": training_names_full,
        "training_names_full_no_rw": training_names_full_no_rw,
        "training_names_full_first_only": training_names_full_first_only,
        "baseline_run": baseline_run,
        "flops_per_step": dict(FLOPS_PER_STEP),
        "clrs_dict_full": dict(CLRS_REGRESSION),
        "loss_histories": loss_histories,
        "eval_data_no_rw": eval_data_no_rw,
        "eval_data_first_only": eval_data_first_only,
        "eval_data_scaling": eval_data_scaling,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--flag",
        "-f",
        default=DEFAULT_WANDB_TAG,
        help="Wandb displayName tag filter (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output root under repo (default: {DEFAULT_OUT_DIR})",
    )
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--suppress-errors", action="store_true")
    p.add_argument(
        "--skip-unevaluated",
        action="store_true",
        help="Skip wandb-tagged runs that do not yet have test_results npz "
        "files on disk (useful when new runs are tagged but evaluate_single_gpu "
        "has not finished). Also enables --suppress-errors for regression.",
    )
    p.add_argument(
        "--regression-only",
        action="store_true",
        help="Only collect regression eval data and write the regression pickle; skip classification.",
    )
    p.add_argument(
        "--classification-only",
        action="store_true",
        help="Only collect classification eval data and write the classification pickle; skip regression.",
    )
    p.add_argument(
        "--skip-wandb",
        action="store_true",
        help="Do not download validation loss histories from wandb. Reuses "
        "loss_histories from an existing pickle for this flag when present.",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path(DEFAULT_DATA_PATH),
        help="Preprocessed dataset root for reco-baseline val CE (default: %(default)s)",
    )
    args = p.parse_args(argv)

    out_dir = _REPO_ROOT / (args.out_dir or DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    clf_path, reg_path = _default_paths(out_dir, args.flag)

    if not args.regression_only:
        print("Collecting classification…")
        clf = collect_classification(
            args.ckpt_dir,
            args.flag,
            data_path=args.data_path,
            skip_wandb=args.skip_wandb,
            skip_unevaluated=args.skip_unevaluated,
            existing_pickle=clf_path if args.skip_wandb else None,
        )
        with open(clf_path, "wb") as f:
            pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Wrote", clf_path)
    else:
        print("Skipping classification (--regression-only).")

    if not args.classification_only:
        print("Collecting regression…")
        reg = collect_regression(
            args.ckpt_dir,
            args.flag,
            args.suppress_errors or args.skip_unevaluated,
            data_path=args.data_path,
            skip_wandb=args.skip_wandb,
            skip_unevaluated=args.skip_unevaluated,
            existing_pickle=reg_path if args.skip_wandb else None,
        )
        with open(reg_path, "wb") as f:
            pickle.dump(reg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Wrote", reg_path)
    else:
        print("Skipping regression (--classification-only).")


if __name__ == "__main__":
    main()
