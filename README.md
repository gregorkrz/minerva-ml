# minerva-ml

This repository contains the data processing and model training code used for ML studies on MINERvA events.

> **Evaluation plots** — browse the latest model-comparison figures (classification, regression, training curves) by configuration:
> **[minerva-ml plots](https://d1to0n5578l1po.cloudfront.net/index.html)**

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

### Dataset distribution plots

After splitting, summarize the ML-ready `.pb` tensors with `src.scripts.plot_dataset_distributions`. Run from the repository root with your project Python environment activated (see section 0).

**Full run** (reads the split, writes all PDFs and a JSON summary):

```bash
cd ~/minerva-ml-hyperscale

python -m src.scripts.plot_dataset_distributions \
  --data-path <SPLIT_OUTPUT_DIR> \
  --playlist 1A \
  --split train \
  --out-dir plots/dataset_distributions
```

**Quick check** on a subsample (faster; good for testing):

```bash
python -m src.scripts.plot_dataset_distributions \
  --data-path <SPLIT_OUTPUT_DIR> \
  --playlist 1A --split train \
  --max-events 50000 \
  --out-dir plots/dataset_distributions
```

#### What gets produced

Outputs live under `plots/dataset_distributions/` (per playlist + split, tag like `1A_train`):


| Path                              | Contents                                           |
| --------------------------------- | -------------------------------------------------- |
| `particle_features_<tag>.pdf`     | Per-particle columns (η, φ, log pT, log E, PID, …) |
| `logE_by_PID_<tag>.pdf`           | log(E) histogram per reconstruction PID type       |
| `global_features_<tag>.pdf`       | 16 event-level conditioning features               |
| `truth_labels_<tag>.pdf`          | All 15 MC truth columns                            |
| `n_particles_per_event_<tag>.pdf` | Token multiplicity                                 |
| `summary_<tag>.json`              | min/max/mean/std/percentiles per variable          |
| `poster/`                         | Publication-style step histograms (see below)      |
| `poster_v2/`                      | Same figures, two-red palette + Matter font        |


**Poster figures** (`poster/` and `poster_v2/`):


| File                                | Description                                                                            |
| ----------------------------------- | -------------------------------------------------------------------------------------- |
| `tokens_per_event_<tag>.pdf`        | Tokens per MINERvA event                                                               |
| `energy_comparison_<tag>.pdf`       | True neutrino energy vs sum of token energies                                          |
| `e_available_spectrum_<tag>.pdf`    | MC `E_available` and `E_available + E_μ`                                               |
| `e_available_by_class_<tag>.pdf`    | `E_ν` plus `E_available` per Pi_labels_v2 class (CC 1π±, CC Nπ±, CC 1π⁰, CC other, NC) |
| `joint/tokens_per_event_<tag>.pdf`  | MINERvA vs JetClass-II token multiplicity overlay                                      |
| `joint/energy_comparison_<tag>.pdf` | MINERvA neutrino energy vs JetClass-II jet energy (log–log)                            |


Poster plots use density-normalized step histograms. The first full run also writes pickle caches under `poster/poster_bins_<tag>.pkl` (and `poster_bins_jetclass2_<tag>.pkl` when JetClass-II is enabled).

#### Fast replot (poster only, no dataset read)

After a full run has built the poster caches, restyle or tweak poster code without reloading `.pb` files:

```bash
python -m src.scripts.plot_dataset_distributions \
  --playlist 1A --split train \
  --out-dir plots/dataset_distributions \
  --poster-from-cache
```

This regenerates both `poster/` and `poster_v2/`. If a cache predates newer plots (e.g. `E_available` spectra), the script reloads truth labels from the dataset only for those missing sections.

#### JetClass-II overlays (optional)

Joint MINERvA/JetClass-II figures need preprocessed JetClass-II h5 shards. By default one file is sampled from the OmniLearned path; skip with `--no-jetclass2`:

```bash
python -m src.scripts.plot_dataset_distributions \
  --data-path <SPLIT_OUTPUT_DIR> \
  --playlist 1A --split train \
  --jetclass2-path /path/to/jetclass2/train \
  --jetclass2-n-files 1 \
  --max-events 50000 \
  --out-dir plots/dataset_distributions
```

#### Useful flags


| Flag                  | Purpose                                        |
| --------------------- | ---------------------------------------------- |
| `--playlist`          | Playlist subfolder (`1A`, `1B`, …)             |
| `--split`             | `train`, `val`, or `test`                      |
| `--max-events`        | Subsample events (deterministic with `--seed`) |
| `--bins`              | Histogram bins for overview grids (default 80) |
| `--log-y`             | Log y-axis on overview continuous histograms   |
| `--poster-from-cache` | Regenerate poster PDFs from cached bins only   |
| `--no-jetclass2`      | Skip JetClass-II and joint overlays            |


#### Publish

Dataset distribution PDFs are included when you publish the local `plots/` tree:

```bash
bash scripts/publish_plots.sh
```

Browse at **[minerva-ml plots](https://d1to0n5578l1po.cloudfront.net/index.html)** → `dataset_distributions`.

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

Run or preview eval for checkpoints that still need `test_results` (skipped if an `.npz` for that dataset already exists). Activate your project Python environment first; child evals use the same interpreter (`sys.executable`).

```bash
# Preview commands
python -m src.scripts.evaluate_single_gpu --wandb-flag <TAG> --dry-run

# Run evals sequentially on the current node (login or single-GPU session)
python -m src.scripts.evaluate_single_gpu --wandb-flag <TAG>
```

`--wandb-flag` only considers runs whose checkpoint folder name matches a wandb run name with that tag; omit it to evaluate every run under `--ckpt-dir` (default: see `--help`). **Evaluation is very small and fast**—it is fine to run on **login nodes** when a GPU is available locally.

#### Multi-GPU evaluation (`--num-gpus`)

On a multi-GPU node, pass `--num-gpus N` (alias `-n N`) to spread the eval jobs across `N` GPUs. One job runs per GPU concurrently, each pinned via `CUDA_VISIBLE_DEVICES`, and idle GPUs immediately pick up the next pending job. Eval jobs are deduplicated the same way as the single-GPU path (a `(folder, dataset)` is skipped if its `.npz` already exists), so the GPUs never repeat work.

```bash
# Request a 4-GPU interactive node (note: --gpus 4)
salloc --nodes 1 --qos interactive --time 04:00:00 --constraint gpu --gpus 4 --account m3246

# Distribute eval jobs across all 4 GPUs
python -m src.scripts.evaluate_single_gpu --wandb-flag <TAG> --num-gpus 4
```

When `CUDA_VISIBLE_DEVICES` is already set (as SLURM does on an allocation), the script only schedules onto those allocated GPUs and caps `--num-gpus` to the number visible (warning if you ask for more). `--num-gpus 1` (the default) keeps the original sequential behavior.

### Evaluation plots (`src.eval`)

Offline plotting reads cached pickles and writes PDFs under `plots/`. Run from the repository root with your project Python environment activated (see section 0).

W&B runs are selected by the **wandb tag** passed as `--flag` (see `fetch_runs_from_wandb` in `src/utils/utils.py`). Model keys in plot configs must match the grouped model names in those pickles (e.g. `OmniLearned-small`, `BERT-tiny`).

#### 1. Cache eval inputs (once per W&B tag)

```bash
cd ~/minerva-ml-hyperscale
export FLAG=Run_2703
export OUT=/global/cfs/cdirs/m3246/gregork/Minerva/runs

python -m src.eval.collect_eval_data \
    --flag $FLAG --out-dir $OUT
```

Writes `classification_<TAG>.pkl` and `regression_<TAG>.pkl` under `$OUT`. Alternatively, download pre-built pickles from [gregorkrzmanc/minerva-ml-eval](https://huggingface.co/datasets/gregorkrzmanc/minerva-ml-eval) and point `$OUT` at that directory.

#### 2. Build plot caches (once; expensive step)

Use `**scripts/rebuild_plot_caches.sh**` to rebuild all CFS caches after new eval data:

```bash
bash scripts/rebuild_plot_caches.sh --force --sync-canonical
```

Caches live on CFS at `**/global/cfs/cdirs/m3246/gregork/Minerva/runs/plots/tmp_results/**` (not in the local repo).


| Cache file                   | Built by                                | Used for                         |
| ---------------------------- | --------------------------------------- | -------------------------------- |
| `classification_metrics.pkl` | `build_classification_cache`            | q3, W, pions (`--plots-only`)    |
| `classification_light.pkl`   | `plot_classification_light`             | light appendix + q3 light panels |
| `steps.pkl`                  | `build_steps_cache`                     | training-curve plots (`plot_steps --plots-only`) |
| `regression.pkl`             | `plot_regression` / `collect_eval_data` | energy-regression plots          |


#### 3. Generate PDFs from cache (fast; config-driven)

**One-shot** (rebuild caches + render + publish):

```bash
bash scripts/rebuild_plot_caches.sh --force --config default --publish
```

**PDFs only** (caches already fresh):

```bash
bash scripts/generate_comparison_plots.sh --config default
bash scripts/publish_plots.sh
```

#### 4. Publish plots (S3 / CloudFront)

Plot PDFs are not tracked in git. After generating locally, publish the HTML index and figures:

```bash
bash scripts/publish_plots.sh
```

Public browse URL: **[https://d1to0n5578l1po.cloudfront.net/index.html](https://d1to0n5578l1po.cloudfront.net/index.html)**

**Single config** (example: BERT vs OmniLearned):

```bash
CONFIG=plot_configs/bert_vs_ol.json
PLOTS=plots/bert_vs_ol
CACHE=plots/tmp_results/classification_metrics.pkl

python -m src.eval.plot_classification_q3 \
    --plots-only --config $CONFIG --plots-dir $PLOTS --metrics-cache $CACHE
python -m src.eval.plot_classification_W \
    --plots-only --config $CONFIG --plots-dir $PLOTS --metrics-cache $CACHE
python -m src.eval.plot_classification_Pions \
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

- `**name**` — must match a model key in the eval pickles (from W&B grouping).
- `**color**` — matplotlib color for that model.
- `**display_name**` — optional legend label override.
- `**step_cutoff**` — optional int; clips the x-axis on log-steps plots only.

**Existing configs:** `default.json` (full model lineup), `bert_vs_ol.json`, `hyperscale.json`, `ol_vs_hyperscale.json`, `V1Paper.json`, `20260606_Comparison.json`.

**Add a new configuration:**

1. Copy an existing file, e.g. `cp plot_configs/bert_vs_ol.json plot_configs/my_paper.json`.
2. Edit the `models` list — keep only the models you want on the figures; reuse colors from `default.json` or `src/eval/_constants.py` (`CLRS_CLASSIFICATION` / `CLRS_REGRESSION`) for consistency.
3. Run plots into a matching output folder:
  ```bash
   python -m src.eval.plot_classification_q3 \
       --plots-only --config plot_configs/my_paper.json --plots-dir plots/my_paper \
       --metrics-cache plots/tmp_results/classification_metrics.pkl
  ```
   Or add `my_paper.json` to the loop in `scripts/generate_comparison_plots.sh` (it already picks up every `plot_configs/*.json`).

Output layout per config: `plots/<config>/classification/{q3,w_bins,pions,light}/`, `plots/<config>/regression/`, `plots/<config>/steps_combined/`, etc.

#### Full recompute (no cache)

If caches are missing or eval data changed, run the individual `plot_*` scripts **without** `--plots-only` (slower; recomputes and refreshes caches):

```bash
python -m src.eval.plot_steps              --flag $FLAG --out-dir $OUT
python -m src.eval.plot_regression             --flag $FLAG --out-dir $OUT
python -m src.eval.plot_classification_W      --flag $FLAG --out-dir $OUT
python -m src.eval.plot_classification_q3     --flag $FLAG --out-dir $OUT
python -m src.eval.plot_classification_Pions  --flag $FLAG --out-dir $OUT
python -m src.eval.plot_classification_light  --flag $FLAG --out-dir $OUT
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

## 7) Small per-bin demo datasets (event viewer + scoring)

A lightweight demo pipeline that carves the ML-ready dataset into small
per-kinematic-bin samples, renders an interactive HTML event viewer, and scores
those samples with trained models. Useful for inspecting event displays
alongside per-model output scores. All three steps below are quick and run on a
single GPU (scoring) or CPU (extraction / viewer).

### a) Generate the small dataset

`src.scripts.extract_bin_demo_datasets` samples a few signal and background
events per bin and saves each *(task, bin, signal|background)* as its own `.pb`
dataset in the **same format** as the main dataset (`data`, `truth_labels`,
`global_features`; see [DATASET.md](DATASET.md)). Bins match the classification
plots:

- **CCNπ± (N≥1)** binned in true hadronic *W* (`DEFAULT_W_BIN_EDGES_GEV`),
- **CC1π±** and **CC1π⁰** binned in true pion energy (equal-frequency edges from
  the true signal).

Bins with no signal or no background are skipped.

```bash
python -m src.scripts.extract_bin_demo_datasets \
  --output-dir /global/cfs/cdirs/m3246/gregork/Minerva/20260326_NEW_DEMO_ONLY
```

Defaults: `--data-dir /global/cfs/cdirs/m3246/gregork/Minerva/20260326_NEW`,
`--playlist 1A`, `--split test`, `--n-events 10`, `--n-pion-bins 5`, `--seed 42`.
Output layout: `<task>/<bin>/<signal|background>/0.pb` (+ a `meta.json` per
dataset and a top-level `manifest.json`). Each `0.pb` loads directly with
`HEPTorchDataset(folder=...)`.

### b) Build the interactive event viewer

`src.scripts.make_event_viewer` reads the demo datasets and writes a single
self-contained HTML page (Plotly from CDN) with selectors for signal
definition / bin / class and per-event 3D + η–φ displays.

```bash
python -m src.scripts.make_event_viewer \
  --input-dir /global/cfs/cdirs/m3246/gregork/Minerva/20260326_NEW_DEMO_ONLY
```

Defaults: `--input-dir .../20260326_NEW_DEMO_ONLY`, `--output plots/event_viewer.html`.
The output is a single fully self-contained HTML file (Plotly from CDN): open the
local `plots/event_viewer.html` in a browser, or view the published copy:

**▶ [Live event viewer](https://d1to0n5578l1po.cloudfront.net/event_viewer.html)**

(`plots/` is gitignored and published to CloudFront via
[`scripts/publish_plots.sh`](scripts/publish_plots.sh); run that to refresh the
hosted page after regenerating.) If model scores are present (step **c**), the
viewer also shows a toggleable per-event score table.

### c) Evaluate models on the small dataset

`src.scripts.eval_demo_datasets` loads **every checkpoint matching a wandb tag**
and scores **every** demo dataset, writing per-model output scores. Run on a GPU
session (e.g. `salloc … --gpus 1`) with W&B access (`WANDB_ENTITY` / `wandb login`).

```bash
python -m src.scripts.eval_demo_datasets \
  --input-dir /global/cfs/cdirs/m3246/gregork/Minerva/20260326_NEW_DEMO_ONLY \
  --flag <TAG> --use-amp
```

Defaults: `--ckpt-dir /global/cfs/cdirs/m3246/gregork/checkpoints`,
`--batch-size 512`, auto device. Use `--runs <name ...>` to score explicit
checkpoints instead of resolving from `--flag`.

Outputs:

- `<task>/<bin>/<class>/scores/<run>.npz` next to each `0.pb` — `prediction`
  (per-class softmax probabilities for classifiers, or regressed energy for
  regression checkpoints; interpret classifier columns via `class_idx`),
  `logits`, and MC-truth `pid`. Event order matches `0.pb` / `meta.json`.
- `<input-dir>/scores.json` — combined index with a `models` section (human-
  readable model name, seed, mode, `num_classes`, `class_idx`) and
  `scores[task/bin/class][run].prediction`.

## Citation

If you use this code or dataset, please cite:

```bibtex
@misc{krzmanc2026,
      title={Cross-Domain Transfer with Particle Physics Foundation Models: From Jets to Neutrino Interactions},
      author={Gregor Krzmanc and Vinicius Mikuni and Benjamin Nachman and Callum Wilkinson},
      year={2026},
      eprint={2604.12364},
      archivePrefix={arXiv},
      primaryClass={hep-ex},
      url={https://arxiv.org/abs/2604.12364},
}
```

See also [CITATION.bib](CITATION.bib).

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

