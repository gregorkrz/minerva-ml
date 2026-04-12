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

Offline plotting reads cached pickles under `out/eval_data/` (by default) and writes PDFs under `plots/` (by default). Run from the repository root with `PYTHONPATH` set to the repo (or install the package in editable mode) so imports resolve.

**1. Cache eval inputs** (loads checkpoints / W&B histories; paths are configurable in `src/eval/_constants.py`):

```bash
export PYTHONPATH="$PWD"
python -m src.eval.collect_eval_data --flag <TAG>
```

This writes `out/eval_data/classification_<TAG>.pkl` and `out/eval_data/regression_<TAG>.pkl`. Use `--out-dir` optionally.

**2. Generate PDFs** (each script accepts `--flag`, `--out-dir`, and `--plots-dir`; defaults match the layout below):

```bash
python -m src.eval.plot_steps              # training curves → plots/{regression,classification}/steps/
python -m src.eval.plot_regression         # energy / q₃ / scaling → plots/regression/
python -m src.eval.plot_classification_W  # vs hadronic W → plots/classification/w_bins/
python -m src.eval.plot_classification_q3  # vs q₃, CCNπ, light appendix → plots/classification/q3/ and .../light/
python -m src.eval.plot_classification_Pions  # pion kinematics, CC1π⁰, light appendix → plots/classification/pions/ and .../light/
```

**3. Figures for a LaTeX paper** (copies or single-page extracts into `figures_latex/`):

```bash
python -m src.scripts.copy_figures_for_paper
# or: python -m src.scripts.copy_figures_for_paper --dry-run
```

Use `--plots-root` if your PDFs live outside the default `plots/` tree (`--out-root` is an alias). Outputs go under `figures_latex/regression`, `figures_latex/classification`, and `figures_latex/classification_detailed` by default.

### Notebooks (optional)

The `notebooks/Eval_*.ipynb` notebooks remain useful for interactive exploration. Set `WANDB_TAG` at the top to match your tag and run all cells. To execute headlessly (requires `nbconvert`; e.g. `pip install nbconvert`):

```bash
cd notebooks
# Execute (updates notebooks in place with fresh outputs)
jupyter nbconvert --to notebook --execute Eval_Regression.ipynb --inplace
jupyter nbconvert --to notebook --execute Eval_Classification.ipynb --inplace
jupyter nbconvert --to notebook --execute Eval_Classification_Light.ipynb --inplace

# Export static copies (HTML; use --to pdf instead if pandoc/LaTeX are available, or --to webpdf with Chromium)
jupyter nbconvert --to html Eval_Regression.ipynb
jupyter nbconvert --to html Eval_Classification.ipynb
```

For publication-quality PDFs and the directory layout used by `copy_figures_for_paper`, prefer the `src.eval` pipeline above rather than notebook output paths.

## 6) Event displays

```bash
python -m src.scripts.make_event_displays \
  --input_file <PATH_TO_ROOT_FILE> \
  --output_dir <PATH_TO_OUTPUT_DIR> \
  --n_events 10
```
