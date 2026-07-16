import os
import re
from collections import defaultdict

# Kinematic-binned classifier runs (see submit_binned_loss_ccn1pipm.py).
_BINNED_CLASSIFIER_RE = re.compile(
    r"_classifier_(Transformer[^_]+)_data_cap_(-?\d+)_seed_(\d+)_binned(W|q3)_([A-Za-z0-9]+)",
    re.IGNORECASE,
)
_TRANSFORMER_RAW_TO_PLOT_KEY = {
    "Transformer1": "Transformer-xsmall",
    "Transformer1NR": "Transformer-xsmall",
    "Transformer3NR": "Transformer-small",
    "Transformer2": "Transformer2",
    "Transformer3": "Transformer3",
}
_BINNED_SIGNAL_TO_PLOT_SUFFIX = {
    "ccn1pipm": "Weigh1",
    "ccn1pipmbin": "Weigh2",
    "cc1pipm": "WeighCC1pipm",
    "cc1pi0": "WeighCC1pi0",
    "ccnpipm": "WeighCCNpipm",
}

# ``--predict-baseline`` classifier runs (submit_predict_baseline_ccnpi.py).
_PREDICT_BASELINE_TRANSFORMER_RE = re.compile(
    r"_classifier_(Transformer[^_]+)_data_cap_(-?\d+)_seed_(\d+)_predictBaseline",
    re.IGNORECASE,
)
_PREDICT_BASELINE_MLP_RE = re.compile(
    r"_cond_only_lowLR_(MLP\d+)_classifier_predictBaseline_NR_full_seed(-?\d+)_",
    re.IGNORECASE,
)
# Cond-only MLP: --predict-baseline + kinematic binned loss (see train.py -name suffix).
_PREDICT_BASELINE_MLP_BINNED_RE = re.compile(
    r"_cond_only_lowLR_(MLP\d+)_classifier_predictBaseline_binned(W|q3)_([A-Za-z0-9]+)_NR_full_seed(-?\d+)_",
    re.IGNORECASE,
)
_PREDICT_BASELINE_MLP_BINNED_PLOT_KEY = {
    "w": "MLP-predictBaseline-binnedW",
    "q3": "MLP-predictBaseline-binnedq3",
}

# Cond-only MLP: MC-truth labels + kinematic binned loss (no --predict-baseline).
_MLP_BINNED_RE = re.compile(
    r"_cond_only_lowLR_(MLP\d+)_classifier_binned(W|q3)_([A-Za-z0-9]+)_NR_full_seed(-?\d+)_",
    re.IGNORECASE,
)
_MLP_BINNED_PLOT_KEY: dict[str, dict[str, str]] = {
    "w": {
        "ccn1pipm": "MLP-binnedW",
        "ccn1pipmbin": "MLP-binnedW-Bin",
    },
    "q3": {
        "ccn1pipm": "MLP-binnedq3",
        "ccn1pipmbin": "MLP-binnedq3-Bin",
    },
}


_HYPERSCALE_RUN_RE = re.compile(
    r"_HyperScale_(small|medium)_(?:(rw)_)?(\w+?)_(regression|classifier)_(-?\d+)_",
    re.IGNORECASE,
)


def parse_hyperscale_model_cap(name: str, *, task: str) -> tuple[str, int] | None:
    """Map a HyperScale run name to ``(plot_model_key, data_cap)`` for *task*."""
    m = _HYPERSCALE_RUN_RE.search(name)
    if m is None:
        return None
    size, rw, _variant, run_task, cap_s = m.groups()
    if run_task != task:
        return None
    key = f"HyperScale-{size.lower()}"
    if rw:
        key += "-rw"
    return key, int(cap_s)


def parse_binned_classifier_model_cap(name: str) -> tuple[str, int] | None:
    """Map a binned-loss classifier run name to ``(plot_model_key, data_cap)``.

    Example::

        Run_1703_classifier_Transformer1_data_cap_-1_seed_55_binnedW_CCN1pipm_...
        -> ("Transformer-xsmall-Weigh1", -1)

        Run_..._binnedW_CCN1pipmBin_... -> ("Transformer-xsmall-Weigh2", -1)
    """
    m = _BINNED_CLASSIFIER_RE.search(name)
    if m is None:
        return None
    raw, cap_s, _seed, _var, signal = m.groups()
    base = _TRANSFORMER_RAW_TO_PLOT_KEY.get(raw, raw)
    suffix = _BINNED_SIGNAL_TO_PLOT_SUFFIX.get(signal.lower(), f"binned-{signal}")
    return f"{base}-{suffix}", int(cap_s)


def parse_binned_mlp_model_cap(name: str) -> tuple[str, int] | None:
    """Map cond-only MLP binned-loss runs (MC truth) to plot keys.

    Examples::

        Run_cond_only_lowLR_MLP3_classifier_binnedW_CCN1pipm_NR_full_seed55_...
        -> ("MLP-binnedW", -1)

        Run_..._binnedW_CCN1pipmBin_... -> ("MLP-binnedW-Bin", -1)
    """
    m = _MLP_BINNED_RE.search(name)
    if m is None:
        return None
    _mlp_variant, binned_var, signal, _seed = m.groups()
    plot_key = _MLP_BINNED_PLOT_KEY.get(binned_var.lower(), {}).get(signal.lower())
    if plot_key is None:
        return None
    return plot_key, -1


def parse_predict_baseline_binned_mlp_model_cap(
    name: str,
) -> tuple[str, int] | None:
    """Map cond-only MLP ``--predict-baseline`` + binned-loss runs to plot keys.

    Example::

        Run_cond_only_lowLR_MLP3_classifier_predictBaseline_binnedW_CCN1pipmBin_NR_full_seed55_...
        -> ("MLP-predictBaseline-binnedW", -1)
    """
    m = _PREDICT_BASELINE_MLP_BINNED_RE.search(name)
    if m is None:
        return None
    _mlp_variant, binned_var, _signal, _seed = m.groups()
    plot_key = _PREDICT_BASELINE_MLP_BINNED_PLOT_KEY.get(binned_var.lower())
    if plot_key is None:
        return None
    return plot_key, -1


def parse_predict_baseline_classifier_model_cap(
    name: str,
) -> tuple[str, int] | None:
    """Map a ``--predict-baseline`` classifier run to ``(plot_model_key, data_cap)``.

    Examples::

        Run_1703_classifier_Transformer3NR_data_cap_-1_seed_55_predictBaseline_...
        -> ("Transformer-small-Baseline", -1)

        Run_cond_only_lowLR_MLP3_classifier_predictBaseline_NR_full_seed55_...
        -> ("MLP-Baseline", -1)
    """
    binned_mlp = parse_predict_baseline_binned_mlp_model_cap(name)
    if binned_mlp is not None:
        return binned_mlp
    m = _PREDICT_BASELINE_TRANSFORMER_RE.search(name)
    if m is not None:
        raw, cap_s, _seed = m.groups()
        base = _TRANSFORMER_RAW_TO_PLOT_KEY.get(raw, raw)
        return f"{base}-Baseline", int(cap_s)
    m = _PREDICT_BASELINE_MLP_RE.search(name)
    if m is not None:
        return "MLP-Baseline", -1
    return None


def fetch_runs_from_wandb(tag: str, project: str = "minerva-models") -> set[str]:
    # Helper function to fetch wandb run names from a project with a given tag
    """Fetch wandb run names from project with the given tag."""
    import wandb

    print("Calling wandb API...")
    api = wandb.Api(timeout=60)
    entity = os.environ.get("WANDB_ENTITY") or getattr(api, "default_entity", None)
    if not entity:
        raise RuntimeError(
            "Wandb entity unknown. Set WANDB_ENTITY or log in with wandb login."
        )
    path = f"{entity}/{project}"
    print(f"  Fetching runs from {path} with tag {tag!r}...")
    runs = api.runs(path, filters={"tags": {"$in": [tag]}})
    names = {run.name for run in runs}
    print(f"  Wandb API finished: {len(names)} run(s) found with tag {tag!r}.")
    return names


def get_regression_runs_from_wandb(
    tag: str, project: str = "minerva-models"
) -> set[str]:
    """Get all regression runs from wandb."""
    return fetch_runs_from_wandb(tag, project)


def get_runs_by_model_and_cap(
    tag: str, project: str = "minerva-models"
) -> dict[str, dict[int, list[str]]]:
    """
    Build a dict: model_name -> {dataset_cap -> [run_names]}.

    Run name formats (from submit_train_jobs):
      - OLS_RW: Run_1203_OLS_RW_regression_<cap>_seed...
      - OLS_int: Run_*_OLS_int_regression_<cap>_seed... -> "OmniLearned-small-int" (interactions on)
      - OLS:    Run_1203_OLS_regression_<cap>_seed...
      - OLM:    Run_1203_OLM_FB_regression_<cap>_seed... (legacy: _OLM_regression_)
      - Transformer1 / Transformer1NR: both map to "Transformer-xsmall"
        (Run_*_regression_Transformer1_data_cap_... and ...Transformer1NR_data_cap_...)
      - MLP: Run_cond_only_lowLR_full_seed<SEED>_...
      - MLP (NR full): names containing _cond_only_lowLR_NR_full_seed<SEED>_ or
          _cond_only_lowLR_MLP<N>_NR_full_seed<SEED>_ (MLP sweep / generate_cond_only_jobs).
          Same global ablation as Transformer1NR (--zero-cond-feature 2). Cap is always -1.
          Names with _cond_only_lowLR_classifier_NR_full_ or _MLP<N>_classifier_NR_full_ are
          classification jobs and are excluded here so regression eval only sees regression MLP runs.
      - BERT-tiny / BERT-tiny-rw: Run_*_BERT_tiny_rw_regression_<cap>_seed... and
          Run_*_BERT_tiny_regression_<cap>_seed... (``tiny_rw`` before ``tiny`` so names match correctly).
      - BERT-tiny-energy-order: Run_*_BERT_tiny_energy_order_regression_<cap>_seed...
          (pretrained + --bert-energy-order).
      - Transformer2 DIS-only: names containing ..._regression_Transformer2_data_cap_<cap>_..._DIS_only_...
          map to model key ``Transformer2-DIS`` (all-events Transformer2 stays ``Transformer2``).
      - HyperScale: Run_*_HyperScale_{small|medium}[_rw]_{basic|embedding|pool}_regression_<cap>_...
          -> ``HyperScale-small``, ``HyperScale-small-rw``, etc. (match ``_rw`` before pretrained).
      - BDT regression: Run_*_BDT_regression_... → ``BDT`` (cap -1).
    dataset_cap is -1 for full 6M dataset, or a positive int for smaller caps.
    """
    run_names = fetch_runs_from_wandb(tag, project)
    # model_name -> dataset_cap -> list of run names
    result = defaultdict(lambda: defaultdict(list))
    for name in run_names:
        parsed = regression_model_cap_from_name(name)
        if parsed is not None:
            model, cap = parsed
            result[model][cap].append(name)
    return {model: dict(caps) for model, caps in result.items()}


def regression_model_cap_from_name(name: str) -> tuple[str, object] | None:
    """Map a single regression run name to ``(plot_model_key, dataset_cap)``.

    Pure function (no wandb); returns ``None`` when the name is not a recognized
    regression run. See :func:`get_runs_by_model_and_cap` for the formats.
    """
    model, cap = None, None
    # Match OLS_RW before OLS
    m = re.search(r"_OLS_RW_regression_(-?\d+)_", name)
    if m:
        model, cap = "OmniLearned-small-rw", int(m.group(1))
    if model is None:
        m = re.search(r"_OLS_regression_(-?\d+)_", name)
        if m:
            model, cap = "OmniLearned-small", int(m.group(1))
    if model is None:
        m = re.search(r"_OLM_FB_regression_(-?\d+)_", name)
        if m:
            model, cap = "OmniLearned-medium", int(m.group(1))
    if model is None:
        m = re.search(r"_regression_(Transformer[^_]+)_data_cap_(-?\d+)_", name)
        if m:
            raw, cap = m.group(1), int(m.group(2))
            if "_DIS_only_" in name and raw == "Transformer2":
                model = "Transformer2-DIS"
            elif raw in ("Transformer1", "Transformer1NR"):
                model = "Transformer-xsmall"
            elif raw == "Transformer3NR":
                model = "Transformer-small"
            else:
                model = raw
    if model is None:
        m = re.search(r"_BERT_tiny_energy_order_regression_(-?\d+)_", name)
        if m:
            model, cap = "BERT-tiny-energy-order", int(m.group(1))
    if model is None:
        m = re.search(r"_BERT_tiny_rw_regression_(-?\d+)_", name)
        if m:
            model, cap = "BERT-tiny-rw", int(m.group(1))
    if model is None:
        m = re.search(r"_BERT_tiny_regression_(-?\d+)_", name)
        if m:
            model, cap = "BERT-tiny", int(m.group(1))
    if model is None:
        parsed = parse_hyperscale_model_cap(name, task="regression")
        if parsed is not None:
            model, cap = parsed
    if model is None:
        m = re.search(r"_BDT_regression_", name)
        if m:
            model, cap = "BDT", -1
    if model is None:
        # Sweep-style regression NR-full: ..._cond_only_lowLR_MLP3_NR_full_seed61_...
        # (must not match ..._MLP3_classifier_NR_full_...).
        m = re.search(r"_cond_only_lowLR_(MLP\d+)_NR_full_seed(-?\d+)_", name)
        if m:
            model, cap = "MLP", -1
    if model is None:
        # Legacy regression NR-full only; do not match _cond_only_lowLR_classifier_NR_full_.
        m = re.search(r"_cond_only_lowLR_(?!classifier_)NR_full_seed(-?\d+)_", name)
        if m:
            model, cap = "MLP", -1
    if model is None:
        m = re.search(r"_cond_only_lowLR_([a-zA-Z0-9]+)_seed(-?\d+)_", name)
        if m:
            model = "MLP"
            dscap = m.group(1)
            if dscap == "full":
                cap = -1
            else:
                try:
                    cap = int(dscap)
                except ValueError:
                    cap = dscap
    if model is None:
        return None
    return model, cap


def get_classification_runs_by_model_and_cap(
    tag: str, project: str = "minerva-models"
) -> dict[str, dict[int, list[str]]]:
    """
    Build a dict: model_name -> {dataset_cap -> [run_names]}.

    Run name formats (from submit_train_jobs):
      - OLS_RW: Run_1203_OLS_RW_classifier_<cap>_seed...
      - OLS_int: Run_*_OLS_int_classifier_<cap>_seed... -> "OmniLearned-small-int" (interactions on)
      - OLS:    Run_1203_OLS_classifier_<cap>_seed...
      - OLM:    Run_1203_OLM_FB_classifier_<cap>_seed... (legacy: _OLM_classifier_)
      - Transformer1 / Transformer1NR: both map to "Transformer-xsmall"
        (Run_*_classifier_Transformer1_data_cap_... and ...Transformer1NR_data_cap_...)
      - MLP (standard): substring _class_cond_only_lowLR_<dscap>_seed<SEED>_
          e.g. Run_*_class_cond_only_lowLR_full_seed42_...  -> cap -1 when dscap is "full"
      - MLP (NR full): substring _cond_only_lowLR_(classifier_)NR_full_seed<SEED>_ — the word
          classifier is optional so both Run_*_cond_only_lowLR_NR_full_seed... and
          Run_*_cond_only_lowLR_classifier_NR_full_seed... (classifier job template) match.
          "NR" matches training with --zero-cond-feature 2 (same global ablation as
          Transformer1NR; see generate_cond_only_jobs.py). cap is always -1.
      - BERT-tiny / BERT-tiny-rw: Run_*_BERT_tiny_rw_classifier_<cap>_seed... and
          Run_*_BERT_tiny_classifier_<cap>_seed... (match ``tiny_rw`` before ``tiny``).
      - BERT-tiny-energy-order: Run_*_BERT_tiny_energy_order_classifier_<cap>_seed...
      - Transformer2 DIS-only: ..._classifier_Transformer2_data_cap_<cap>_..._DIS_only_... → ``Transformer2-DIS``.
      - Binned loss: ``..._classifier_Transformer1_data_cap_<cap>_seed_<N>_binnedW_CCN1pipm_...``
        → ``Transformer-xsmall-Weigh1`` (5-class head; see :func:`parse_binned_classifier_model_cap`).
      - Binned binary CCN1pipm: ``..._binnedW_CCN1pipmBin_...`` → ``Transformer-xsmall-Weigh2``.
      - Predict baseline (--predict-baseline): ``..._predictBaseline`` on Transformer runs
        → ``Transformer-small-Baseline`` (etc.); MLP sweep names with
        ``_classifier_predictBaseline_`` → ``MLP-Baseline``.
      - Predict baseline + binned loss (cond-only MLP):
        ``..._classifier_predictBaseline_binnedW_CCN1pipmBin_NR_full_seed...``
        → ``MLP-predictBaseline-binnedW`` (``binnedq3`` → ``MLP-predictBaseline-binnedq3``).
      - Binned loss on MC truth (cond-only MLP):
        ``..._classifier_binnedW_CCN1pipm_NR_full_seed...`` → ``MLP-binnedW``;
        ``..._binnedW_CCN1pipmBin_...`` → ``MLP-binnedW-Bin``.
      - HyperScale: Run_*_HyperScale_{small|medium}[_rw]_..._classifier_<cap>_... (same keys as regression).
    dataset_cap is -1 for full 6M dataset, or a positive int for smaller caps.
    """
    run_names = fetch_runs_from_wandb(tag, project)
    # model_name -> dataset_cap -> list of run names
    result = defaultdict(lambda: defaultdict(list))

    for name in run_names:
        parsed = classification_model_cap_from_name(name)
        if parsed is not None:
            model, cap = parsed
            result[model][cap].append(name)

    return {model: dict(caps) for model, caps in result.items()}


def classification_model_cap_from_name(name: str) -> tuple[str, object] | None:
    """Map a single classification run name to ``(plot_model_key, dataset_cap)``.

    Pure function (no wandb); returns ``None`` when the name is not a recognized
    classification run. See :func:`get_classification_runs_by_model_and_cap`.
    """
    binned = parse_binned_classifier_model_cap(name)
    if binned is not None:
        return binned
    binned_mlp = parse_binned_mlp_model_cap(name)
    if binned_mlp is not None:
        return binned_mlp
    pred_bl = parse_predict_baseline_classifier_model_cap(name)
    if pred_bl is not None:
        return pred_bl
    model, cap = None, None
    # Match OLS_RW before OLS
    m = re.search(r"_OLS_RW_classifier_(-?\d+)_", name)
    if m:
        model, cap = "OmniLearned-small-rw", int(m.group(1))
    if model is None:
        m = re.search(r"_OLS_classifier_(-?\d+)_", name)
        if m:
            model, cap = "OmniLearned-small", int(m.group(1))
    if model is None:
        m = re.search(r"_OLM_FB_?classifier_(-?\d+)_", name)
        if m:
            model, cap = "OmniLearned-medium", int(m.group(1))
    if model is None:
        m = re.search(r"_classifier_(Transformer[^_]+)_data_cap_(-?\d+)_", name)
        if m:
            raw, cap = m.group(1), int(m.group(2))
            if "_DIS_only_" in name and raw == "Transformer2":
                model = "Transformer2-DIS"
            elif raw in ("Transformer1", "Transformer1NR"):
                model = "Transformer-xsmall"
            elif raw == "Transformer3NR":
                model = "Transformer-small"
            else:
                model = raw
    if model is None:
        m = re.search(r"_BERT_tiny_energy_order_classifier_(-?\d+)_", name)
        if m:
            model, cap = "BERT-tiny-energy-order", int(m.group(1))
    if model is None:
        m = re.search(r"_BERT_tiny_rw_classifier_(-?\d+)_", name)
        if m:
            model, cap = "BERT-tiny-rw", int(m.group(1))
    if model is None:
        m = re.search(r"_BERT_tiny_classifier_(-?\d+)_", name)
        if m:
            model, cap = "BERT-tiny", int(m.group(1))
    if model is None:
        parsed = parse_hyperscale_model_cap(name, task="classifier")
        if parsed is not None:
            model, cap = parsed
    if model is None:
        # Cond-only gradient-boosted-tree baseline (train.py --use-bdt).
        # MC-truth BDT with W-binned loss (name contains "binnedW") → "BDT-binnedW";
        # plain MC-truth BDT → "BDT". --predict-baseline runs → "BDT-BC" /
        # "BDT-BC-binnedW".
        m = re.search(r"_BDT_classifier_", name)
        if m:
            if re.search(r"predictBaseline", name, re.IGNORECASE):
                if re.search(r"binnedW", name, re.IGNORECASE):
                    model, cap = "BDT-BC-binnedW", -1
                else:
                    model, cap = "BDT-BC", -1
            elif re.search(r"binnedW", name, re.IGNORECASE):
                model, cap = "BDT-binnedW", -1
            else:
                model, cap = "BDT", -1
    if model is None:
        m = re.search(
            r"_cond_only_lowLR_(MLP\d+)_classifier_NR_full_seed(-?\d+)_", name
        )
        if m:
            model, cap = "MLP", -1
    if model is None:
        m = re.search(r"_cond_only_lowLR_classifier_NR_full_seed(-?\d+)_", name)
        if m:
            model, cap = "MLP", -1
    if model is None:
        m = re.search(r"_class_cond_only_lowLR_([a-zA-Z0-9]+)_seed(-?\d+)_", name)
        if m:
            model = "MLP"
            dscap = m.group(1)
            if dscap == "full":
                cap = -1
            else:
                try:
                    cap = int(dscap)
                except ValueError:
                    cap = dscap
    if model is None:
        return None
    return model, cap
