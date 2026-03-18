import os
import re
from collections import defaultdict


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

def get_regression_runs_from_wandb(tag: str, project: str = "minerva-models") -> set[str]:
    """Get all regression runs from wandb."""
    return fetch_runs_from_wandb(tag, project)


def get_runs_by_model_and_cap(
    tag: str, project: str = "minerva-models"
) -> dict[str, dict[int, list[str]]]:
    """
    Build a dict: model_name -> {dataset_cap -> [run_names]}.

    Run name formats (from submit_train_jobs):
      - OLS_RW: Run_1203_OLS_RW_regression_<cap>_seed...
      - OLS:    Run_1203_OLS_regression_<cap>_seed...
      - OLM:    Run_1203_OLM_regression_<cap>_seed...
      - Transformer: Run_1203_regression_Transformer1_data_cap_<cap>_seed_...
      - MLP: Run_cond_only_full_seed<SEED>_...
    dataset_cap is -1 for full 6M dataset, or a positive int for smaller caps.
    """
    run_names = fetch_runs_from_wandb(tag, project)
    # model_name -> dataset_cap -> list of run names
    result = defaultdict(lambda: defaultdict(list))
    for name in run_names:
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
            m = re.search(r"_OLM_regression_(-?\d+)_", name)
            if m:
                model, cap = "OmniLearned-medium", int(m.group(1))
        if model is None:
            m = re.search(r"_regression_(Transformer\d+)_data_cap_(-?\d+)_", name)
            if m:
                model, cap = m.group(1), int(m.group(2))
                model = "Transformer"
        if model is None:
            m = re.search(r"_cond_only_([a-zA-Z0-9]+)_seed(-?\d+)_", name)
            if m:
                model = "MLP"
                dscap = m.group(1)
                # Determine cap: map "full" to -1, else try to interpret as int if possible
                if dscap == "full":
                    cap = -1
                else:
                    try:
                        cap = int(dscap)
                    except ValueError:
                        cap = dscap  # If not int, just pass the DSCAP string
        if model is not None:
            result[model][cap].append(name)
    return {model: dict(caps) for model, caps in result.items()}

def get_classification_runs_by_model_and_cap(
    tag: str, project: str = "minerva-models"
) -> dict[str, dict[int, list[str]]]:
    """
    Build a dict: model_name -> {dataset_cap -> [run_names]}.

    Run name formats (from submit_train_jobs):
      - OLS_RW: Run_1203_OLS_RW_classifier_<cap>_seed...
      - OLS:    Run_1203_OLS_classifier_<cap>_seed...
      - OLM:    Run_1203_OLS_classifier_<cap>_seed...
      - Transformer: Run_1203_classifier_Transformer1_data_cap_<cap>_seed_...
      - MLP: Run_class_cond_only_<dscap>_seed<SEED>_...
    dataset_cap is -1 for full 6M dataset, or a positive int for smaller caps.
    """
    run_names = fetch_runs_from_wandb(tag, project)
    # model_name -> dataset_cap -> list of run names
    result = defaultdict(lambda: defaultdict(list))

    for name in run_names:
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
            m = re.search(r"_OLM_classifier_(-?\d+)_", name)
            if m:
                model, cap = "OmniLearned-medium", int(m.group(1))
        if model is None:
            m = re.search(r"_classifier_(Transformer\d+)_data_cap_(-?\d+)_", name)
            if m:
                model, cap = m.group(1), int(m.group(2))
                model = "Transformer"
        if model is None:
            m = re.search(r"_class_cond_only_([a-zA-Z0-9]+)_seed(-?\d+)_", name)
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
        if model is not None:
            result[model][cap].append(name)

    return {model: dict(caps) for model, caps in result.items()}
