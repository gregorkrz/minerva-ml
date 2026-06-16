"""Hyperparameter configs for cond-only MLP sweep (MLP1–MLP10)."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_DATA_PATH = "/global/cfs/cdirs/m3246/gregork/Minerva/20260326"
DEFAULT_BATCH_SIZE = 32_000
DEFAULT_D_MODEL = 128
DEFAULT_MAX_STEPS = 256_000
DEFAULT_WARMUP_STEPS = 200
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_EVENT_SAMPLER_SEED = 42
DEFAULT_SEEDS = (50, 51, 52, 53)
DEFAULT_ZERO_COND_FEATURE = 2
DEFAULT_NUM_WORKERS = 16


@dataclass(frozen=True)
class MLPSweepConfig:
    key: str
    mlp_layers: int
    dropout: float
    lr: float

    @property
    def display_label(self) -> str:
        return (
            f"{self.key} (L={self.mlp_layers}, drop={self.dropout:g}, lr={self.lr:g})"
        )


MLP_SWEEP_CONFIGS: dict[str, MLPSweepConfig] = {
    "MLP1": MLPSweepConfig("MLP1", mlp_layers=4, dropout=0.0, lr=1e-4),
    "MLP2": MLPSweepConfig("MLP2", mlp_layers=2, dropout=0.0, lr=1e-4),
    "MLP3": MLPSweepConfig("MLP3", mlp_layers=3, dropout=0.0, lr=1e-4),
    "MLP4": MLPSweepConfig("MLP4", mlp_layers=6, dropout=0.0, lr=1e-4),
    "MLP5": MLPSweepConfig("MLP5", mlp_layers=4, dropout=0.05, lr=1e-4),
    "MLP6": MLPSweepConfig("MLP6", mlp_layers=4, dropout=0.15, lr=1e-4),
    "MLP7": MLPSweepConfig("MLP7", mlp_layers=4, dropout=0.0, lr=1e-3),
    "MLP8": MLPSweepConfig("MLP8", mlp_layers=4, dropout=0.0, lr=1e-5),
    "MLP9": MLPSweepConfig("MLP9", mlp_layers=2, dropout=0.1, lr=1e-3),
    "MLP10": MLPSweepConfig("MLP10", mlp_layers=6, dropout=0.1, lr=1e-5),
}

MLP_SWEEP_ORDER: tuple[str, ...] = tuple(f"MLP{i}" for i in range(1, 11))


def run_name(cfg: MLPSweepConfig, *, task: str, seed: int) -> str:
    """W&B run name prefix (train.py appends a timestamp)."""
    if task == "regression":
        return f"Run_cond_only_lowLR_{cfg.key}_NR_full_seed{seed}"
    if task == "classifier":
        return f"Run_cond_only_lowLR_{cfg.key}_classifier_NR_full_seed{seed}"
    raise ValueError(f"Unknown task: {task!r}")


def build_train_cmd(
    cfg: MLPSweepConfig,
    *,
    task: str,
    seed: int,
    data_path: str = DEFAULT_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_steps: int = DEFAULT_MAX_STEPS,
    warmup_steps: int = DEFAULT_WARMUP_STEPS,
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM_STEPS,
    event_sampler_seed: int = DEFAULT_EVENT_SAMPLER_SEED,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> str:
    """Return a single ``python -m src.scripts.train ...`` command."""
    name = run_name(cfg, task=task, seed=seed)
    common = (
        f"python -m src.scripts.train -bs {batch_size} "
        f"--d_model {DEFAULT_D_MODEL} --mlp_layers {cfg.mlp_layers} "
        f"--dropout {cfg.dropout} --cond_only --seed {seed} "
        f"-seed-event-sampler {event_sampler_seed} "
        f"--num_workers {num_workers} "
        f"--max_steps {max_steps} --warmup_steps {warmup_steps} "
        f"--grad_accum_steps {grad_accum_steps} "
        f"--lr {cfg.lr:g} --data_path {data_path} "
        f"--zero-cond-feature {DEFAULT_ZERO_COND_FEATURE} "
        f"-name {name}"
    )
    if task == "regression":
        return f"{common} --mode regression -E-available-no-muon"
    if task == "classifier":
        return f"{common} --mode classifier -npi2"
    raise ValueError(f"Unknown task: {task!r}")


def mlp_sweep_flops_per_step(mlp_layers: int, *, baseline_layers: int = 4) -> float:
    """Scale baseline MLP FLOPs (2.6e9 at 4 layers) by residual depth."""
    return 2.6 * 1e9 * (mlp_layers / baseline_layers)
