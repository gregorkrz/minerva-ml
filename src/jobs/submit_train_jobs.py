# Train the transformer locally
from src.constants.slurm_template import SLURM_TEMPLATE_GPU
import os
from datetime import datetime as dt


# ---------------------------------------------------------------------------
# HyperScale presets (PLACEHOLDERS — fill in before submitting).
#
# - HYPERSCALE_HYPERPARAMS: pick d_model / depth / n_heads / mlp_ratio per size.
#   These map to --d_model / --depth / --n_heads / --hs-mlp-ratio in train.py.
# - HYPERSCALE_PRETRAINED_PATHS: absolute paths to pretrained checkpoints used
#   by the non-"rw" variants. The "-rw" variants ignore this dict and train
#   from random init.
#
#   The --hs-pretrained flag (added in train.py) loads encoder weights only —
#   the task output head and any global-feature projection are left at random
#   init, so the same checkpoint can be reused across regression / classifier
#   runs with different output dims.
# ---------------------------------------------------------------------------
# Architecture comes from the checkpoint dir's train_config.yaml:
#   model_type: ParticleVIT_Embedding
#   model_params: { embed_dim: 448, depth: 4, num_heads: 7, mlp_ratio: 8/3 }
#
# NOTE: the ckpt below was trained as ParticleVIT_Embedding (split kin/PID/vertex
# input embeddings). For --use-hyperscale embedding the full encoder transfers;
# for basic/pool only the transformer blocks (and CLS/norm where shapes match)
# load, because their token_embed module differs — expect a "loaded X/Y model
# tensors" line with X significantly less than Y in those runs.
HYPERSCALE_HYPERPARAMS = {
    "small": {
        "d_model": 448,
        "depth": 4,
        "n_heads": 7,
        "mlp_ratio": 8 / 3,
    },
}

HYPERSCALE_PRETRAINED_PATHS = {
    "small": (
        "/global/cfs/cdirs/m3246/jaluus/Hyperscale_V5/Pretrain_Scaling/"
        "emb_6e17_d4_e448_bs512_lr5e-4_run3_normalized"
    ),
}

HYPERSCALE_VARIANTS = ("basic", "embedding", "pool")


def _parse_hyperscale_tag(model):
    """Parse 'HyperScale-{size}[-rw]-{variant}' into (size, random_init, variant)."""
    parts = model.split("-")
    # parts[0] = "HyperScale", parts[1] = size, then optionally "rw", then variant.
    if len(parts) < 3 or parts[0] != "HyperScale":
        raise ValueError(f"Invalid HyperScale model tag: {model!r}")
    size = parts[1]
    if size not in HYPERSCALE_HYPERPARAMS:
        raise ValueError(
            f"Unknown HyperScale size {size!r}; "
            f"expected one of {tuple(HYPERSCALE_HYPERPARAMS)}"
        )
    if parts[2] == "rw":
        random_init = True
        variant = "-".join(parts[3:])
    else:
        random_init = False
        variant = "-".join(parts[2:])
    if variant not in HYPERSCALE_VARIANTS:
        raise ValueError(
            f"Unknown HyperScale variant {variant!r}; "
            f"expected one of {HYPERSCALE_VARIANTS}"
        )
    return size, random_init, variant


def get_env_vars():
    """Build `export` lines from `.env` for SLURM, excluding WANDB_* (no API key in job files).

    Training loads repo-root `.env` at runtime via `src.scripts.train`.
    """
    env_commands = ""
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith("WANDB_"):
                continue
            value = value.strip()
            env_commands += f"export {key}={value}\n"
    return env_commands


def generate_cmd(
    data_cap=-1,
    seed=42,
    task="regression",
    model="Transformer1",
    bs=2048,
    max_steps=250000,
    grad_accum_steps=1,
    continue_from="",
    resume_run_id="",
    resume_run_name="",
    model_n_layers=4,
    event_types=None,
):
    if continue_from:
        base = f"python -m src.scripts.train --resume {continue_from} -name {resume_run_name} --resume-run-id {resume_run_id} --max_steps 1000000"
        return base.format(continue_from=continue_from)
    base = "python -m src.scripts.train -bs {bs} --mode {task} {detailed_task} -name {name} --d_model {model_dim} --depth {model_depth} --n_heads {model_n_heads} --dropout {model_dropout} --attn_dropout {model_attn_dropout} {cap} --seed {seed} -seed-event-sampler {seed}  --max_steps {max_steps} --grad_accum_steps {grad_accum_steps} {extra} --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260326 "
    # Model options: ... "OLS", "OLS_int", "OLS_RW", "OLM"/"OLM_FB",
    # "BERT-tiny", "BERT-tiny-rw" (BERT tiny arch, random encoder weights),
    # "HyperScale-small[-rw]-{basic|embedding|pool}" (HyperScale ParticleViT;
    # "-rw" trains from random init, otherwise loads HYPERSCALE_PRETRAINED_PATHS[size]).
    # Seed both the event sampler and the whole training with seed
    detailed_task = "-E-available-no-muon" if task == "regression" else "-npi2"
    model_dim = 128
    model_depth = 4
    model_n_heads = 8
    model_dropout = 0.0
    model_attn_dropout = 0.0
    if model == "OLS":
        name = f"Run_1703_OLS_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned small --use-pretrained pretrain_s --zero-cond-feature 2 "
    elif model == "OLS_int":
        name = f"Run_1703_OLS_int_{task}_{data_cap}_seed{seed}"
        extra = (
            " --use-omnilearned small --use-pretrained pretrain_s --zero-cond-feature 2 "
            "--ol-interaction --ol-local-interaction "
        )
    elif model == "OLS_RW":
        name = f"Run_1703_OLS_RW_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned small --zero-cond-feature 2 "
    elif model == "OLM" or model == "OLM_FB":
        # Medium OmniLearned: unified Run_1703_OLM_FB_{task}_... (classifier uses frozen backbone in train.py).
        name = f"Run_1703_OLM_FB_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned medium --use-pretrained pretrain_m --zero-cond-feature 2 "
    elif model == "BERT-tiny":
        name = f"Run_1703_BERT_tiny_{task}_{data_cap}_seed{seed}"
        extra = " --use-bert tiny --zero-cond-feature 2 "
    elif model == "BERT-tiny-rw":
        name = f"Run_1703_BERT_tiny_rw_{task}_{data_cap}_seed{seed}"
        extra = " --use-bert tiny-rw --zero-cond-feature 2 "
    elif model.startswith("HyperScale-"):
        hs_size, hs_random_init, hs_variant = _parse_hyperscale_tag(model)
        hp = HYPERSCALE_HYPERPARAMS[hs_size]
        model_dim = hp["d_model"]
        model_depth = hp["depth"]
        model_n_heads = hp["n_heads"]
        rw_tag = "_rw" if hs_random_init else ""
        name = (
            f"Run_1703_HyperScale_{hs_size}{rw_tag}_{hs_variant}"
            f"_{task}_{data_cap}_seed{seed}"
        )
        extra = (
            f" --use-hyperscale {hs_variant} "
            f"--hs-mlp-ratio {hp['mlp_ratio']} "
            "--zero-cond-feature 2 "
        )
        if not hs_random_init:
            extra += f" --hs-pretrained {HYPERSCALE_PRETRAINED_PATHS[hs_size]} "
    elif model in (
        "Transformer1",
        "Transformer1NR",
        "Transformer2",
        "Transformer3",
        "Transformer3NR",
    ):
        name = f"Run_1703_{task}_{model}_data_cap_{data_cap}_seed_{seed}"
        # extra = " --zero-cond-feature 2 "
        extra = ""
        if model == "Transformer1NR":
            extra = " --zero-cond-feature 2 "
        elif model == "Transformer2":
            model_n_heads = 12
            model_dim = 384
            model_depth = 8
        elif model == "Transformer3":
            model_dim = 184
            model_depth = 7
            model_n_heads = 8
        elif model == "Transformer3NR":
            model_dim = 184
            model_depth = 7
            model_n_heads = 8
            extra = " --zero-cond-feature 2 "
    else:
        raise ValueError(f"Invalid model: {model}")
    cap = f" -cap {data_cap} " if data_cap > 0 else ""
    # Optional event-type filter (e.g. DIS-only classification runs).
    if event_types:
        types_str = " ".join(str(t) for t in event_types)
        extra = extra + f" --event-types {types_str} "
        # Tag the run name so it doesn't collide with unfiltered runs.
        suffix = "_" + "".join(str(t).upper() for t in event_types) + "_only"
        name = name + suffix
    return base.format(
        bs=bs,
        task=task,
        detailed_task=detailed_task,
        name=name,
        cap=cap,
        seed=seed,
        max_steps=max_steps,
        extra=extra,
        grad_accum_steps=grad_accum_steps,
        model_dim=model_dim,
        model_depth=model_depth,
        model_n_heads=model_n_heads,
        model_dropout=model_dropout,
        model_attn_dropout=model_attn_dropout,
        model_n_layers=model_n_layers,
    )


def get_cmds_and_slurm_times():
    """Transformer-large (= ``Transformer2`` preset, d_model=384, depth=8, n_heads=12),
    trained on DIS events only. Sweeps over seeds and tasks for direct comparison.
    Also submits two BERT-tiny classification and two BERT-tiny regression runs
    (same seeds and training budget as the Transformer2 sweep; all events, no DIS filter).

    NOTE: the previous BERT-tiny + DIS-only extra run sweep is kept below as a
    commented reference — uncomment to restore it.
    """
    cmds = []
    slurm_times = []
    seeds = [55, 56]
    tasks = ["regression", "classifier"]
    model = "Transformer2"  # Transformer-large
    # Transformer2 is ~10x the params of Transformer1/BERT-tiny so per-step compute
    # dominates; max_steps is unchanged by the event-types filter. Walltime below is
    # a conservative default — tune after first job finishes.
    walltime = "20:00:00"
    for seed in seeds:
        for task in tasks:
            cmd = generate_cmd(
                data_cap=-1,
                seed=seed,
                task=task,
                model=model,
                max_steps=500000,
                bs=2048,
                grad_accum_steps=1,
                event_types=["DIS"],
            )
            cmds.append(cmd)
            slurm_times.append(walltime)
    # BERT-tiny: 2× regression + 2× classification (seeds 55, 56), all events.
    walltime_bert = "05:00:00"
    for seed in seeds:
        for task in tasks:
            cmd = generate_cmd(
                data_cap=-1,
                seed=seed,
                task=task,
                model="BERT-tiny",
                max_steps=500000,
                bs=2048,
                grad_accum_steps=1,
            )
            cmds.append(cmd)
            slurm_times.append(walltime_bert)
    return cmds, slurm_times


def get_cmds_and_slurm_times_hyperscale():
    """HyperScale sweep: small × {pretrained, rw} × embedding × 4 seeds ×
    {regression, classifier} = 16 jobs.

    Restricted to the ``embedding`` variant because the pretrained checkpoint
    at HYPERSCALE_PRETRAINED_PATHS["small"] is a ParticleVIT_Embedding run —
    the basic / pool variants would only partially transfer (token_embed shape
    mismatch). Add them back to ``variants`` once matching checkpoints exist.
    """
    cmds = []
    slurm_times = []
    seeds = [55, 56, 57, 58]
    tasks = ["regression", "classifier"]
    sizes = ["small"]
    variants = ["embedding"]
    # TODO: tune per-config after the first job finishes (pretrained vs rw
    # typically have different convergence budgets).
    walltime = "12:00:00"
    for size in sizes:
        for variant in variants:
            for use_rw in (False, True):
                rw_part = "rw-" if use_rw else ""
                model_tag = f"HyperScale-{size}-{rw_part}{variant}"
                for seed in seeds:
                    for task in tasks:
                        cmd = generate_cmd(
                            data_cap=-1,
                            seed=seed,
                            task=task,
                            model=model_tag,
                            max_steps=500000,
                            bs=2048,
                            grad_accum_steps=1,
                        )
                        cmds.append(cmd)
                        slurm_times.append(walltime)
    return cmds, slurm_times


# ---------------------------------------------------------------------------
# Previous sweep — kept for reference. Uncomment (and rename) to restore.
# BERT-tiny over (seed, task) plus one extra DIS-only classification run.
# ---------------------------------------------------------------------------
# def get_cmds_and_slurm_times_bert_tiny():
#     times_data_cap = {  # for 20k: for OLS_RW 100 minutes, for the rest OLS 50 minutes
#         "20000": {
#             "OLS_RW": "01:30:00",
#             "OLS": "00:50:00",
#             "OLS_int": "00:50:00",
#             "OLM_FB": "00:50:00",
#             "BERT-tiny": "00:50:00",
#             "BERT-tiny-rw": "00:50:00",
#             "Transformer1": "00:20:00",
#             "Transformer1NR": "00:20:00",
#         },
#         "50000": {
#             "OLS_RW": "03:00:00",
#             "OLS": "01:00:00",
#             "OLS_int": "01:00:00",
#             "OLM_FB": "01:00:00",
#             "BERT-tiny": "01:00:00",
#             "BERT-tiny-rw": "01:00:00",
#             "Transformer1": "00:20:00",
#             "Transformer1NR": "00:20:00",
#         },
#         "100000": {
#             "OLS_RW": "04:00:00",
#             "Transformer1": "00:20:00",
#             "Transformer1NR": "00:20:00",
#             "OLS": "01:00:00",
#             "OLS_int": "01:00:00",
#             "OLM_FB": "01:00:00",
#             "BERT-tiny": "01:00:00",
#             "BERT-tiny-rw": "01:00:00",
#         },
#         "200000": {
#             "OLS_RW": "05:00:00",
#             "OLS": "01:30:00",
#             "OLS_int": "01:30:00",
#             "OLM_FB": "01:30:00",
#             "BERT-tiny": "01:30:00",
#             "BERT-tiny-rw": "01:30:00",
#             "Transformer1": "00:20:00",
#             "Transformer1NR": "00:20:00",
#         },
#     }
#     cmds = []
#     slurm_times = []
#     for seed in [55, 56]:
#         for data_cap in [-1]:
#             for task in ["regression", "classifier"]:
#                 for model in ["BERT-tiny"]:  # use "BERT-tiny-rw" for same arch, random BERT weights
#                     if "OL" in model:
#                         bs = 2048
#                         grad_accum_steps = 1
#                         if "OLM_FB" in model or "OLM" in model:
#                             bs = 512
#                             grad_accum_steps = 4
#                             if data_cap == -1:
#                                 slurm_times.append("15:00:00")
#                             else:
#                                 slurm_times.append(times_data_cap[str(data_cap)][model])
#                         else:
#                             if data_cap == -1:
#                                 slurm_times.append("12:00:00")
#                             else:
#                                 slurm_times.append(times_data_cap[str(data_cap)][model])
#                     elif model in ("BERT-tiny", "BERT-tiny-rw"):
#                         bs = 2048
#                         grad_accum_steps = 1
#                         if data_cap == -1:
#                             slurm_times.append("12:00:00")
#                         else:
#                             slurm_times.append(times_data_cap[str(data_cap)][model])
#                     else:
#                         bs = 2048
#                         grad_accum_steps = 1
#                         if task == "regression":
#                             if data_cap == -1:
#                                 slurm_times.append("08:00:00")
#                             else:
#                                 slurm_times.append(times_data_cap[str(data_cap)][model])
#                         else:
#                             if data_cap == -1:
#                                 slurm_times.append("05:00:00")
#                             else:
#                                 slurm_times.append(times_data_cap[str(data_cap)][model])
#                     max_steps = {
#                         "regression": 500000,
#                         "classifier": 500000,
#                     }
#                     cmd = generate_cmd(
#                         data_cap=data_cap,
#                         seed=seed,
#                         task=task,
#                         model=model,
#                         max_steps=max_steps[task],
#                         bs=bs,
#                         grad_accum_steps=grad_accum_steps,
#                     )
#                     cmds.append(cmd)
#
#     # Extra: one DIS-only classification run (BERT-tiny) to investigate poor
#     # performance of the all-events model on DIS events. Matches the sweep's
#     # BERT-tiny classifier config (bs, grad_accum, max_steps, walltime).
#     cmds.append(
#         generate_cmd(
#             data_cap=-1,
#             seed=55,
#             task="classifier",
#             model="BERT-tiny",
#             max_steps=500000,
#             bs=2048,
#             grad_accum_steps=1,
#             event_types=["DIS"],
#         )
#     )
#     slurm_times.append("12:00:00")
#     return cmds, slurm_times


def get_cmds_and_slurm_times_continue():
    # Use this to continue runs that were cut short. TODO: change run names and wandb IDs
    """to_resume = [
        "Run_1703_OLM_regression_-1_seed50_20260328_115708",
        "Run_1703_OLS_RW_regression_-1_seed50_20260327_174607"
    ]
    names = [
        "Run_1703_OLM_regression_-1_seed50",
        "Run_1703_OLS_RW_regression_-1_seed50"
    ]
    run_ids = [
        "ofpk105k",
        "rd9azt5b"
    ]"""
    to_resume = [
        "Run_1703_OLM_classifier_-1_seed51_20260331_151803",
        "Run_1703_OLS_RW_classifier_-1_seed50_20260329_201841",
        "Run_1703_OLM_classifier_-1_seed50_20260330_172545",
        "Run_1703_OLS_RW_regression_-1_seed51_20260330_201339",
        "Run_1703_OLM_regression_-1_seed51_20260330_220131",
        "Run_1703_OLS_RW_classifier_-1_seed51_20260330_234443",
        "Run_1703_OLS_RW_regression_-1_seed52_20260330_235131",
        "Run_1703_OLS_RW_classifier_-1_seed52_20260331_014742",
        "Run_1703_OLS_RW_regression_-1_seed53_20260331_022931",
        "Run_1703_OLM_classifier_-1_seed51_20260331_151803",
        "Run_1703_OLM_regression_-1_seed52_20260331_164052",
        "Run_1703_OLM_classifier_-1_seed52_20260401_010518",
        "Run_1703_OLM_regression_-1_seed53_20260401_033613",
        "Run_1703_OLS_RW_classifier_-1_seed53_20260401_050053",
    ]
    names = [
        "Run_1703_OLM_classifier_-1_seed51",
        "Run_1703_OLS_RW_classifier_-1_seed50",
        "Run_1703_OLM_classifier_-1_seed50",
        "Run_1703_OLS_RW_regression_-1_seed51",
        "Run_1703_OLM_regression_-1_seed51",
        "Run_1703_OLS_RW_classifier_-1_seed51",
        "Run_1703_OLS_RW_regression_-1_seed52",
        "Run_1703_OLS_RW_classifier_-1_seed52",
        "Run_1703_OLM_classifier_-1_seed51",
        "Run_1703_OLS_RW_classifier_-1_seed53",
        "Run_1703_OLM_regression_-1_seed53",
        "Run_1703_OLS_RW_classifier_-1_seed53",
        "Run_1703_OLM_classifier_-1_seed53",
        "Run_1703_OLS_RW_classifier_-1_seed53",
    ]
    run_ids = [
        "kdzhg3i3",
        "kdzhg3i3",
        "qwmm0fhb",
        "3vpr9i0q",
        "3vpr9i0q",
        "u5zspjc5",
        "jrb9mx10",
        "vyb2ys44",
        "j1fg2w1i",
        "kdzhg3i3",
        "45tqa47o",
        "k6dp4wc1",
        "2n7dxql3",
        "fgarftde",
    ]
    CKPT_DIR = "/global/cfs/cdirs/m3246/gregork/checkpoints"
    cmds = []
    slurm_times = []
    for i, ckpt in enumerate(to_resume):
        ckpt_path = os.path.join(CKPT_DIR, ckpt, "best_model.pt")
        cmd = generate_cmd(
            continue_from=ckpt_path, resume_run_name=names[i], resume_run_id=run_ids[i]
        )
        cmds.append(cmd)
        slurm_times.append("12:00:00")
    return cmds, slurm_times


CONTAINER_IMAGE = "docker.io/gkrz/minerva_ml:v1"

if __name__ == "__main__":
    # To submit the HyperScale sweep instead (16 jobs across small ×
    # pretrained/rw × embedding × 4 seeds × {regression, classifier}),
    # swap the next line for:
    #     cmds, slurm_times = get_cmds_and_slurm_times_hyperscale()
    cmds, slurm_times = get_cmds_and_slurm_times()
    for i, cmd in enumerate(cmds):
        job_name = f"run_{i}_{dt.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = (
            f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/run_100326/{job_name}.log"
        )
        error_dir = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/run_100326/{job_name}.error.log"
        slurm_file = (
            f"/global/cfs/cdirs/m3246/gregork/Minerva/slurm/run_100326/{job_name}.slurm"
        )
        os.makedirs(os.path.dirname(slurm_file), exist_ok=True)
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)
        with open(slurm_file, "w") as f:
            f.write(
                SLURM_TEMPLATE_GPU.format(
                    queue_name="shared",
                    time=slurm_times[i],
                    cpus_per_task=32,
                    gpus_per_node=1,
                    job_name=job_name,
                    log_dir=log_dir,
                    error_dir=error_dir,
                    commands=cmd,
                    env_vars=get_env_vars(),
                    container_image=CONTAINER_IMAGE,
                )
            )
        print(f"Saved slurm file to {slurm_file}")
        os.system(f"sbatch {slurm_file}")
        print("Job submitted")
