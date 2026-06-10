# minerva-ml

This repository contains the data processing and model training code used for ML studies on MINERvA events.

## Dataset

The processed dataset used by this project is available on Hugging Face:

- [gregorkrzmanc/minerva-ml](https://huggingface.co/datasets/gregorkrzmanc/minerva-ml)
- It is a preprocessed version of the MINERvA open data release for ML/physics tasks such as available-energy estimation and event tagging.
- Source data comes from MINERvA open data: [MINERvA Open Data](https://minerva.fnal.gov/opendata/).
- This is a derived dataset and is not an official MINERvA collaboration product.

For detailed data fields and semantics, see **[DATASET.md](DATASET.md)**.
For model architecture details, see **[MODELS.md](MODELS.md)**.

## Repository workflow

The typical workflow is:

1. Download raw playlists
2. Preprocess ROOT files into ML-ready tensors
3. Split into train/val/test
4. Train models locally or submit SLURM jobs
5. Run test evaluation (`eval`) on checkpoints, then produce figures with `src.eval` (or notebooks)

## 0) Environment set-up

Use the `gkrz/minerva_ml:v1` container (see the provided `Dockerfile`).


## 1) Get the data (two options)

Choose one of the following:

- **Option A (from scratch):** Download raw MINERvA playlists, then preprocess locally.
- **Option B (quick start):** Download the already preprocessed dataset from Hugging Face and skip local preprocessing.

### Option A: Download raw playlists (from scratch)

Set `SCRATCH` first, then run:

```bash
# Monte Carlo playlists
python -m src.scripts.download_data

# Recorded data playlists
python -m src.scripts.download_data --prefix MediumEnergy_FHC_Data_Playlist
```

### Option B: Download preprocessed dataset from Hugging Face

If you want to skip raw playlist processing, download the preprocessed dataset snapshot:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download gregorkrzmanc/minerva-ml \
  --repo-type dataset \
  --local-dir <HF_DATA_DIR>
```

After download, point your training/splitting commands to the downloaded folder structure.

## 2) Preprocess dataset

Skip this section if you used **Option B** and already have the preprocessed files you need.

Minimal invocation (creates `.pb` files with event-wise particle tensors and labels):

```bash
python -m src.scripts.preprocess_dataset --output-dir <OUTPUT_DIR>
```

For a full pipeline on this project’s layout—preprocess, split playlists `1A` / `1B`, and extract baselines—edit paths in the script if needed, then run:

```bash
bash src/scripts/preprocess.sh
```

`src/scripts/preprocess.sh` sets `DATA_DIR`, runs `preprocess_dataset` with blob/prong limits and playlist selection, runs `split_dataset` per playlist (with different val/test ratios for `1B` vs `1A`), and runs `extract_baselines` against the raw playlist directories under scratch.

## 3) Split into train / val / test

```bash
python -m src.scripts.split_dataset \
  --input-dir <PREPROCESSED_DIR> \
  --output-dir <SPLIT_OUTPUT_DIR>
```

To inspect created features quickly, see `notebooks/stats.ipynb`.

## 4) Train models

### Direct training command

Use `src/scripts/train.py` for both regression and classification.

```bash
python -m src.scripts.train \
  -bs 2048 \
  --mode regression \
  -E-available-no-muon \
  -name Run_debug \
  --d_model 128 --depth 4 --n_heads 8 \
  --max_steps 500000 \
  --data_path <SPLIT_OUTPUT_DIR>
```

### SLURM submission (`src/jobs/submit_train_jobs.py`)

`src/jobs/submit_train_jobs.py` builds training commands, writes SLURM scripts, and submits them with `sbatch`.

Current defaults in that script:
- loops over `seed`, `data_cap`, `task`, and `model`
- uses `task in {regression, classifier}` (these map directly to `--mode` values)
- supports `model in {Transformer1, OLS, OLS_RW, OLM}`
- maps each `(data_cap, model)` to a SLURM walltime
- writes `.slurm`, `.log`, and `.error.log` files under fixed NERSC paths

Before running submission:

1. Create a `.env` file in repo root with environment variables needed in your cluster job.
2. Update hardcoded paths in `submit_train_jobs.py` if you are not using the default NERSC layout (for example `--data_path` in `generate_cmd`, `CKPT_DIR` in resume mode, and the `log_dir` / `error_dir` / `slurm_file` paths in `__main__`).
3. Optionally edit `get_cmds_and_slurm_times()` to choose your model/task/data-cap sweep.

Then submit:

```bash
python src/jobs/submit_train_jobs.py
```

The script also includes `get_cmds_and_slurm_times_continue()` for checkpoint resume runs.

## 5) Analysis (test eval, evaluation plots, notebooks)

After training, group the runs you want to compare in [Weights & Biases](https://wandb.ai) by assigning the **same tag** to each run (in the run’s overview or via the API). The tools below use that tag against the `minerva-models` project under your W&B entity (set `WANDB_ENTITY` and use `wandb login` as needed).

### Test evaluation (`eval`)

Generate the `python -m src.scripts.eval ...` commands for checkpoints that still need `test_results` (skipped if an `.npz` for that dataset already exists):

```bash
python -m src.scripts.print_eval_commands --wandb-flag <TAG>
```

`--wandb-flag` only lists runs whose checkpoint folder name matches a wandb run name with that tag; omit it to consider every run under `--ckpt-dir` (default: see `--help`). Run each printed line locally. **Evaluation is very small and fast**—it is fine to run on **login nodes** without a GPU job.

### Evaluation plots (`src.eval`)

Offline plotting reads cached pickles and writes PDFs under `plots/`. Run from the repository root (on Perlmutter, use the `omni` env: `/global/homes/g/gregork/.conda/envs/omni/bin/python`).

W&B runs are selected by the **wandb tag** passed as `--flag` (see `fetch_runs_from_wandb` in `src/utils/utils.py`). Model keys in plot configs must match the grouped model names in those pickles (e.g. `OmniLearned-small`, `BERT-tiny`).

#### 1. Cache eval inputs (once per W&B tag)

```bash
cd ~/minerva-ml-hyperscale
export FLAG=Run_2703
export OUT=/global/cfs/cdirs/m3246/gregork/Minerva/runs

/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.collect_eval_data \
    --flag $FLAG --out-dir $OUT
```

Writes `classification_<TAG>.pkl` and `regression_<TAG>.pkl` under `$OUT`. Alternatively, download pre-built pickles from [gregorkrzmanc/minerva-ml-eval](https://huggingface.co/datasets/gregorkrzmanc/minerva-ml-eval) and point `$OUT` at that directory.

#### 2. Build plot caches (once; expensive step)

These caches live under `plots/tmp_results/` and let later runs use `--plots-only` without reloading the 16 GB classification pickle or recomputing metrics.

```bash
# Classification metrics (q3, W, pion kinematics, per-inttype baselines, …)
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.build_classification_cache \
    --flag $FLAG --out-dir $OUT --force

# Other caches (run once each; also built automatically on a normal non--plots-only run)
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_steps              --flag $FLAG --out-dir $OUT
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_classification_light --flag $FLAG --out-dir $OUT
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_regression         --flag $FLAG --out-dir $OUT
```

| Cache file | Built by | Used for |
|---|---|---|
| `classification_metrics.pkl` | `build_classification_cache` | q3, W, pions (`--plots-only`) |
| `classification_light.pkl` | `plot_classification_light` | light appendix + q3 light panels |
| `steps.pkl` | `plot_steps` | training-curve plots |
| `regression.pkl` | `plot_regression` / `collect_eval_data` | energy-regression plots |

#### 3. Generate PDFs from cache (fast; config-driven)

**Recommended:** `scripts/generate_comparison_plots.sh` loops over every JSON in `plot_configs/` and writes one output tree per config under `plots/<config_name>/`.

```bash
bash scripts/generate_comparison_plots.sh
```

The script builds `classification_metrics.pkl` automatically if it is missing, then runs all `plot_*` modules with `--plots-only` and `--config`. Edit `FLAG`, `OUT_DIR`, and `METRICS_CACHE` at the top of the script if your paths differ.

**Single config** (example: BERT vs OmniLearned):

```bash
CONFIG=plot_configs/bert_vs_ol.json
PLOTS=plots/bert_vs_ol
CACHE=plots/tmp_results/classification_metrics.pkl

/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_classification_q3 \
    --plots-only --config $CONFIG --plots-dir $PLOTS --metrics-cache $CACHE
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_classification_W \
    --plots-only --config $CONFIG --plots-dir $PLOTS --metrics-cache $CACHE
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_classification_Pions \
    --plots-only --config $CONFIG --plots-dir $PLOTS --metrics-cache $CACHE
# … likewise for plot_steps, plot_classification_light, plot_regression
```

#### Plot configs (`plot_configs/`)

Each `*.json` file selects which models appear on the figures and their colors. Schema (see `src/eval/_plot_config.py`):

```json
{
  "models": [
    {"name": "OmniLearned-small", "color": "#ff7f0e", "display_name": "OL-small"},
    {"name": "BERT-tiny", "color": "#0d9488", "display_name": "BERT-small"}
  ],
  "step_cutoff": null
}
```

- **`name`** — must match a model key in the eval pickles (from W&B grouping).
- **`color`** — matplotlib color for that model.
- **`display_name`** — optional legend label override.
- **`step_cutoff`** — optional int; clips the x-axis on log-steps plots only.

**Existing configs:** `default.json` (full model lineup), `bert_vs_ol.json`, `hyperscale.json`, `V1Paper.json`, `20260606_Comparison.json`.

**Add a new configuration:**

1. Copy an existing file, e.g. `cp plot_configs/bert_vs_ol.json plot_configs/my_paper.json`.
2. Edit the `models` list — keep only the models you want on the figures; reuse colors from `default.json` or `src/eval/_constants.py` (`CLRS_CLASSIFICATION` / `CLRS_REGRESSION`) for consistency.
3. Run plots into a matching output folder:
   ```bash
   /global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_classification_q3 \
       --plots-only --config plot_configs/my_paper.json --plots-dir plots/my_paper \
       --metrics-cache plots/tmp_results/classification_metrics.pkl
   ```
   Or add `my_paper.json` to the loop in `scripts/generate_comparison_plots.sh` (it already picks up every `plot_configs/*.json`).

Output layout per config: `plots/<config>/classification/{q3,w_bins,pions,light}/`, `plots/<config>/regression/`, `plots/<config>/steps_combined/`, etc.

#### Full recompute (no cache)

If caches are missing or eval data changed, run the individual `plot_*` scripts **without** `--plots-only` (slower; recomputes and refreshes caches):

```bash
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_steps              --flag $FLAG --out-dir $OUT
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_regression             --flag $FLAG --out-dir $OUT
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_classification_W      --flag $FLAG --out-dir $OUT
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_classification_q3     --flag $FLAG --out-dir $OUT
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_classification_Pions  --flag $FLAG --out-dir $OUT
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.eval.plot_classification_light  --flag $FLAG --out-dir $OUT
```

Plotting code lives under `src.eval.classification_plots` and `src.eval.e_available_plots`; notebooks may use thin shims in `notebooks/`.

#### Figures for a LaTeX paper

Copies or single-page extracts into `figures_latex/`:

```bash
python -m src.scripts.copy_figures_for_paper
# or: python -m src.scripts.copy_figures_for_paper --dry-run
```


## 6) Event displays

```bash
python -m src.scripts.make_event_displays \
  --input_file <PATH_TO_ROOT_FILE> \
  --output_dir <PATH_TO_OUTPUT_DIR> \
  --n_events 10
```
