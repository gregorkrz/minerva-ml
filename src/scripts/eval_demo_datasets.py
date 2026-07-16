#!/usr/bin/env python3
"""Score every per-bin demo dataset with all checkpoints for a wandb flag.

Given the per-bin signal/background datasets produced by
``src/scripts/extract_bin_demo_datasets.py`` (``manifest.json`` plus
``<task>/<bin>/<signal|background>/0.pb`` folders), this loads **every
checkpoint whose wandb run name matches a given flag/tag** and runs inference on
**every** demo dataset, writing per-model output scores.

For each ``(run, dataset)`` it writes ``<dataset>/scores/<run_name>.npz`` with:

* ``prediction`` — class probabilities ``(N, num_classes)`` for classifiers
  (softmax of the logits, matching ``classification_plots.load_results``), or the
  scalar regression output ``(N,)`` for regression checkpoints;
* ``logits`` — the raw model outputs (classifiers only);
* ``pid``     — MC-truth Pi_labels_v2 class per event (aligned, shuffle=False).

It also writes a combined ``<input-dir>/scores.json`` indexing every
``task/bin/class`` → ``run`` → rounded predictions, for easy demo consumption.
Event order matches ``0.pb`` / ``meta.json`` (no shuffling), so ``prediction[i]``
corresponds to event ``i`` in the viewer.

Progress is cached in ``<input-dir>/scores_cache.json`` (same ``{models,
scores}`` shape, accumulated across flags/invocations and keyed by run name),
rewritten after each successfully-scored run. On a re-run, any run already
present in the cache is skipped, so you only pay for models that haven't been
scored yet. Pass ``--force`` to ignore the cache and re-evaluate everything.

Run names are resolved from wandb (project ``minerva-models``) by tag, then
intersected with checkpoint folders that contain ``best_model.pt`` — the same
selection logic as ``src/scripts/evaluate_single_gpu.py``.

NOTE: requires a GPU session for any reasonable speed, and wandb access to list
runs for the flag (set ``WANDB_ENTITY`` / ``wandb login``; an ``.env`` at the
repo root is auto-loaded).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset.dataloader import (
    HEPTorchDataset,
    collate_point_cloud,
    get_Pi_labels_v2,
)
from src.eval._constants import plot_model_label
from src.scripts.eval import create_model_from_checkpoint
from src.scripts.train import (
    _bdt_cond_batch,
    _bdt_proba_full,
    forward_model,
    prepare_batch,
    prepare_batch_bert,
    prepare_batch_hyperscale,
    prepare_batch_omnilearned,
)

DEFAULT_INPUT_DIR = Path(
    "/global/cfs/cdirs/m3246/gregork/Minerva/20260326_NEW_DEMO_ONLY"
)
DEFAULT_CKPT_DIR = Path("/global/cfs/cdirs/m3246/gregork/checkpoints")
DEFAULT_FLAG = "Run_2306"


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def parse_seed(run: str) -> int | None:
    """Extract the integer seed from a run name (``seed42`` / ``seed_42``)."""
    m = re.search(r"seed_?(\d+)", run)
    return int(m.group(1)) if m else None


def _fallback_model_name(run: str) -> str:
    """Best-effort human-readable model name from a run name (no wandb needed).

    Mirrors the families parsed by ``src/utils/utils.get_*_runs_by_model_and_cap``.
    """
    r = run
    m = re.search(r"HyperScale_(small|medium)(_rw)?", r)
    if m:
        return f"HyperScale-{m.group(1)}" + ("-rw" if m.group(2) else "")
    if "BERT_tiny_rw" in r:
        return "BERT-tiny-rw"
    if "BERT_tiny_energy_order" in r:
        return "BERT-tiny-energy-order"
    if "BERT_tiny" in r:
        return "BERT-tiny"
    if "OLS_RW" in r:
        return "OmniLearned-small-rw"
    if "OLS_int" in r:
        return "OmniLearned-small-int"
    if re.search(r"\bOLM\b", r) or "_OLM_" in r:
        return "OmniLearned-medium"
    if "OLS" in r:
        return "OmniLearned-small"
    if "Transformer2" in r:
        return "Transformer2-DIS" if "DIS_only" in r else "Transformer2"
    if "Transformer1" in r:
        return "Transformer-xsmall"
    if "cond_only" in r or "_MLP" in r or r.startswith("MLP"):
        return "MLP"
    return run


def resolve_model_names(flag: str, runs: list[str]) -> dict[str, str]:
    """Map each run name to a human-readable model name.

    Prefers the authoritative wandb parsers (``get_*_runs_by_model_and_cap``);
    falls back to :func:`_fallback_model_name` for runs they don't cover or when
    wandb is unavailable.
    """
    run_to_raw: dict[str, str] = {}
    try:
        from src.utils.utils import (
            get_classification_runs_by_model_and_cap,
            get_runs_by_model_and_cap,
        )

        for getter in (
            get_classification_runs_by_model_and_cap,
            get_runs_by_model_and_cap,
        ):
            try:
                model_map = getter(flag)
            except Exception as exc:  # noqa: BLE001
                print(f"  (model-name lookup via {getter.__name__} failed: {exc})")
                continue
            for model, caps in model_map.items():
                for run_list in caps.values():
                    for r in run_list:
                        run_to_raw.setdefault(r, model)
    except Exception as exc:  # noqa: BLE001
        print(f"  (model-name lookup unavailable: {exc})")

    out: dict[str, str] = {}
    for run in runs:
        raw = run_to_raw.get(run)
        out[run] = plot_model_label(raw) if raw else _fallback_model_name(run)
    return out


def resolve_runs(flag: str, ckpt_dir: Path) -> list[str]:
    """Run names with *flag* (wandb) that also exist as checkpoints with best_model.pt."""
    from src.scripts.evaluate_single_gpu import (
        get_all_ckpt_folders,
        get_folders_from_wandb,
    )

    wandb_names = get_folders_from_wandb(flag)
    available = get_all_ckpt_folders(str(ckpt_dir))
    runs = sorted(wandb_names & available)
    missing = sorted(wandb_names - available)
    if missing:
        print(f"  ({len(missing)} run(s) for flag {flag!r} have no checkpoint here)")
    return runs


def enumerate_datasets(input_dir: Path) -> list[tuple[str, Path]]:
    """Return ``(relative_key, abs_dir)`` for every demo dataset folder."""
    manifest_path = input_dir / "manifest.json"
    out: list[tuple[str, Path]] = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        for task in manifest.get("tasks", {}).values():
            for b in task.get("bins", []):
                for cls in ("signal", "background"):
                    if cls in b:
                        rel = b[cls]["path"]
                        d = input_dir / rel
                        if (d / "0.pb").exists():
                            out.append((rel, d))
    else:
        for pb in sorted(input_dir.glob("*/*/*/0.pb")):
            d = pb.parent
            out.append((str(d.relative_to(input_dir)), d))
    return out


def build_loader(
    folder: Path, args_dict: dict, task, batch_size: int, max_particles: int
):
    """Mirror ``load_data``'s HEPTorchDataset/DataLoader build for an arbitrary folder."""
    use_omnilearned = args_dict.get("use_omnilearned", None)
    use_bert = args_dict.get("use_bert", None)
    concat_additional_info = not (bool(use_omnilearned) or bool(use_bert))
    with contextlib.redirect_stdout(io.StringIO()):
        ds = HEPTorchDataset(
            folder=str(folder),
            use_cond=args_dict.get("use_cond", False),
            use_pid=args_dict.get("use_pid", True),
            pid_idx=args_dict.get("pid_idx", 4),
            max_particles=max_particles,
            task=task,
            concat_additional_info=concat_additional_info,
            nevts=-1,
            use_energy_sums=args_dict.get("include_E_sum", False),
        )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=lambda x: collate_point_cloud(x, max_particles=max_particles),
    )
    return loader


@torch.no_grad()
def run_inference(model, loader, device, args_dict, use_amp: bool) -> np.ndarray:
    """Return stacked raw model outputs for the loader (order preserved)."""
    use_omnilearned = args_dict.get("use_omnilearned", None)
    use_bert = args_dict.get("use_bert", None)
    use_hyperscale = args_dict.get("use_hyperscale", None)
    use_cond = args_dict.get("use_cond", False)
    use_pid = args_dict.get("use_pid", True)
    coord_dim = args_dict.get("coord_dim", 2)
    pid_idx = args_dict.get("pid_idx", 4)
    include_E_sum = args_dict.get("include_E_sum", False)
    zero_cond_feature = args_dict.get("zero_cond_feature", None)
    args_ns = SimpleNamespace(**args_dict)
    mode = args_dict.get("mode", "classifier")

    outs = []
    for batch in loader:
        if use_omnilearned:
            inputs = prepare_batch_omnilearned(
                batch, device, use_cond, use_pid, pid_idx, include_E_sum=include_E_sum
            )
        elif use_bert:
            inputs = prepare_batch_bert(
                batch,
                device,
                use_pid=use_pid,
                pid_idx=pid_idx,
                use_cond=use_cond,
                include_E_sum=include_E_sum,
                zero_cond_feature=zero_cond_feature,
                energy_order=args_dict.get("bert_energy_order", False),
            )
        elif use_hyperscale:
            inputs = prepare_batch_hyperscale(
                batch,
                device,
                use_pid=use_pid,
                pid_idx=pid_idx,
                use_cond=use_cond,
                include_E_sum=include_E_sum,
                zero_cond_feature=zero_cond_feature,
                variant=use_hyperscale,
            )
        else:
            inputs = prepare_batch(
                batch,
                device,
                use_cond,
                use_pid,
                coord_dim,
                pid_idx,
                include_E_sum=include_E_sum,
                zero_cond_feature=zero_cond_feature,
            )
        amp_enabled = bool(use_amp and device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = forward_model(model, inputs, args_ns)
        if mode == "regression":
            outs.append(logits.squeeze(-1).float().cpu().numpy())
        else:
            outs.append(logits.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def run_inference_bdt(model, loader, task, args_dict: dict) -> np.ndarray:
    """Score a scikit-learn BDT (``HistGradientBoosting*``) on cond features.

    BDTs aren't callable torch modules, so they can't go through
    :func:`run_inference`. Mirrors the training/eval path (``_bdt_cond_batch`` +
    ``_bdt_proba_full`` / ``.predict``): returns per-class **probabilities**
    ``(N, num_classes)`` for classifiers (already normalized, so the caller must
    not softmax them) or the scalar regression output ``(N,)``.
    """
    # _bdt_cond_batch reads args.include_E_sum / args.zero_cond_feature; ensure
    # both exist so a checkpoint missing them doesn't raise AttributeError.
    args_ns = SimpleNamespace(
        include_E_sum=args_dict.get("include_E_sum", False),
        zero_cond_feature=args_dict.get("zero_cond_feature", None),
        **{k: v for k, v in args_dict.items()
           if k not in ("include_E_sum", "zero_cond_feature")},
    )
    is_reg = task.type == "regression"
    num_classes = len(task.class_idx) if not is_reg else None
    outs = []
    for batch in loader:
        X, _, _ = _bdt_cond_batch(batch, args_ns, with_weights=False)
        if is_reg:
            outs.append(np.asarray(model.predict(X), dtype=np.float64).reshape(-1))
        else:
            outs.append(_bdt_proba_full(model, X, num_classes))
    return np.concatenate(outs, axis=0)


def truth_pid_for_dataset(folder: Path) -> np.ndarray:
    """MC-truth Pi_labels_v2 per event from the dataset's own truth labels."""
    meta = folder / "meta.json"
    if meta.exists():
        with open(meta) as f:
            pid = json.load(f).get("pid_classes")
        if pid is not None:
            return np.asarray(pid, dtype=np.int64)
    blob = torch.load(folder / "0.pb", weights_only=False, map_location="cpu")
    tl = blob["truth_labels"]
    if not torch.is_tensor(tl):
        tl = torch.as_tensor(tl)
    return get_Pi_labels_v2(tl).numpy().astype(np.int64)


def model_summary(args_dict: dict, task) -> dict:
    info = {"mode": task.type}
    if task.type == "classifier":
        info["num_classes"] = len(task.class_idx) if task.class_idx else None
        info["class_idx"] = list(task.class_idx) if task.class_idx else None
        if getattr(task, "binary_classifier", False):
            info["binary_signal_pid_classes"] = list(
                getattr(task, "binary_signal_pid_classes", []) or []
            )
    info["log1p_loss"] = bool(args_dict.get("log1p_loss", False))
    return info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Demo datasets dir (default: {DEFAULT_INPUT_DIR})",
    )
    ap.add_argument(
        "--flag",
        "-f",
        default=DEFAULT_FLAG,
        help=f"wandb tag selecting runs (default: {DEFAULT_FLAG})",
    )
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=DEFAULT_CKPT_DIR,
        help=f"Checkpoint root (default: {DEFAULT_CKPT_DIR})",
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        default=None,
        help="Explicit run names to use instead of resolving from --flag.",
    )
    ap.add_argument("--batch-size", "-bs", type=int, default=512)
    ap.add_argument(
        "--max-runs", type=int, default=None, help="Limit number of runs (debugging)."
    )
    ap.add_argument("--use-amp", action="store_true", help="Mixed precision on CUDA.")
    ap.add_argument("--device", default=None, help="torch device (default: auto).")
    ap.add_argument(
        "--force",
        "--ignore-cache",
        dest="force",
        action="store_true",
        help="Ignore scores_cache.json and re-evaluate every run from scratch.",
    )
    args = ap.parse_args(argv)

    if not args.input_dir.exists():
        sys.exit(f"Input dir not found: {args.input_dir}")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(42)
    print(f"Device: {device}")

    datasets = enumerate_datasets(args.input_dir)
    if not datasets:
        sys.exit(f"No demo datasets found under {args.input_dir}")
    print(f"Found {len(datasets)} demo datasets.")

    # Truth pid per dataset (model-independent), cached once.
    truth_pid = {rel: truth_pid_for_dataset(d) for rel, d in datasets}

    if args.runs:
        runs = list(args.runs)
        print(f"Using {len(runs)} explicit run(s).")
    else:
        print(f"Resolving runs for flag {args.flag!r} …")
        runs = resolve_runs(args.flag, args.ckpt_dir)
    if args.max_runs:
        runs = runs[: args.max_runs]
    if not runs:
        sys.exit("No runs to evaluate.")
    print(f"Evaluating {len(runs)} run(s) over {len(datasets)} datasets.")

    print("Resolving human-readable model names …")
    run_to_model = resolve_model_names(args.flag, runs)

    # Persistent per-run cache (accumulated across flags/invocations, keyed by
    # run name) so already-scored models can be skipped on re-runs. Same
    # {models, scores} shape as scores.json but a superset of runs. It is
    # rewritten after every successfully-scored run so progress survives a crash.
    cache_path = args.input_dir / "scores_cache.json"
    cache: dict = {"models": {}, "scores": {}}
    if cache_path.exists() and not args.force:
        try:
            with open(cache_path) as f:
                loaded = json.load(f)
            cache["models"] = dict(loaded.get("models", {}))
            cache["scores"] = {
                rel: dict(sc) for rel, sc in loaded.get("scores", {}).items()
            }
            print(f"Loaded {cache_path.name} ({len(cache['models'])} model(s) cached).")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not read {cache_path.name} ({exc}); starting fresh.")
            cache = {"models": {}, "scores": {}}
    for rel, _ in datasets:
        cache["scores"].setdefault(rel, {})

    def run_fully_cached(run: str) -> bool:
        """True iff the cache has this run's info and scores for every dataset."""
        if run not in cache["models"]:
            return False
        return all(run in cache["scores"].get(rel, {}) for rel, _ in datasets)

    def write_cache() -> None:
        tmp = cache_path.with_name(cache_path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(cache, f, separators=(",", ":"))
        tmp.replace(cache_path)

    for ri, run in enumerate(runs, start=1):
        if not args.force and run_fully_cached(run):
            print(f"[{ri}/{len(runs)}] {run}: cached, skipping eval")
            continue
        ckpt = args.ckpt_dir / run / "best_model.pt"
        if not ckpt.exists():
            print(f"[{ri}/{len(runs)}] {run}: no best_model.pt, skipping")
            continue
        print(f"[{ri}/{len(runs)}] {run}: loading model …")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model, args_dict, task = create_model_from_checkpoint(str(ckpt), device)
        except Exception as exc:  # noqa: BLE001
            print(f"    failed to load ({exc}); skipping")
            continue

        max_particles = args_dict.get("max_particles", 33)
        info = model_summary(args_dict, task)
        model_name = run_to_model.get(run) or _fallback_model_name(run)
        seed = parse_seed(run)
        info = {
            "model": model_name,
            "seed": seed,
            "label": f"{model_name} (seed {seed})" if seed is not None else model_name,
            "run_name": run,
            **info,
        }
        cache["models"][run] = info
        mode = info["mode"]
        is_bdt = bool(args_dict.get("use_bdt", False))

        try:
            for rel, folder in datasets:
                with contextlib.redirect_stdout(io.StringIO()):
                    loader = build_loader(
                        folder, args_dict, task, args.batch_size, max_particles
                    )
                    if is_bdt:
                        raw = run_inference_bdt(model, loader, task, args_dict)
                    else:
                        raw = run_inference(
                            model, loader, device, args_dict, args.use_amp
                        )
                score_dir = folder / "scores"
                score_dir.mkdir(parents=True, exist_ok=True)
                pid = truth_pid[rel]

                if mode == "classifier":
                    # BDT already returns normalized probabilities; NN returns
                    # logits that still need a softmax.
                    if is_bdt:
                        probs = raw
                        logits = np.log(np.clip(probs, 1e-12, None))
                    else:
                        logits = raw
                        probs = _softmax(logits)
                    np.savez(
                        score_dir / f"{run}.npz",
                        prediction=probs,
                        logits=logits,
                        pid=pid,
                    )
                    cache["scores"][rel][run] = {
                        "mode": mode,
                        "prediction": np.round(probs, 5).tolist(),
                    }
                else:
                    pred = raw
                    if info["log1p_loss"]:
                        pred = np.maximum(np.exp(pred) - 1.0, 0.0)
                    np.savez(score_dir / f"{run}.npz", prediction=pred, pid=pid)
                    cache["scores"][rel][run] = {
                        "mode": mode,
                        "prediction": np.round(pred, 4).tolist(),
                    }
            # Persist progress after each successful run so a later crash (or a
            # bad run) never discards work already done.
            write_cache()
            print(f"    done ({mode}, {len(datasets)} datasets); cache updated.")
        except Exception as exc:  # noqa: BLE001
            # One bad run (e.g. a scikit-learn BDT that isn't a callable torch
            # module) must not abort the whole eval and lose every other run's
            # scores. Drop this run's partial results and keep going.
            print(f"    inference failed ({exc}); skipping this run")
            cache["models"].pop(run, None)
            for rel, _ in datasets:
                cache["scores"][rel].pop(run, None)
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Build scores.json for *this flag* from the cache: only runs that were
    # requested and are fully scored (drops load/inference failures, e.g. BDTs),
    # so the event viewer never renders empty rows.
    scored_runs = [r for r in runs if run_fully_cached(r)]
    scores_json = {
        "flag": args.flag,
        "ckpt_dir": str(args.ckpt_dir),
        "input_dir": str(args.input_dir),
        "runs": scored_runs,
        "models": {r: cache["models"][r] for r in scored_runs},
        "scores": {
            rel: {r: cache["scores"][rel][r] for r in scored_runs}
            for rel, _ in datasets
        },
    }

    out_json = args.input_dir / "scores.json"
    with open(out_json, "w") as f:
        json.dump(scores_json, f, separators=(",", ":"))
    print(
        f"\nWrote per-dataset npz under each <dataset>/scores/, "
        f"{cache_path} (cache), and {out_json}"
    )
    print(
        f"Models scored for flag {args.flag!r}: {len(scored_runs)} "
        f"(cache holds {len(cache['models'])} total)"
    )


if __name__ == "__main__":
    main()
