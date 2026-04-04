# minerva-data-processing

This repository contains the data processing and model training code used for ML studies on MINERvA events.

For detailed data fields and semantics, see **[DATASET.md](DATASET.md)**.
For model architecture details, see **[MODELS.md](MODELS.md)**.

## Repository workflow

The typical workflow is:

1. Download raw playlists
2. Preprocess ROOT files into ML-ready tensors
3. Split into train/val/test
4. Train models locally or submit SLURM jobs
5. Run test evaluation (`eval`) on checkpoints, then analyze with notebooks

## 1) Download data

Set `SCRATCH` first, then run:

```bash
# Monte Carlo playlists
python -m src.scripts.download_data

# Recorded data playlists
python -m src.scripts.download_data --prefix MediumEnergy_FHC_Data_Playlist
```

## 2) Preprocess dataset

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

## 5) Analysis (test eval + notebooks)

After training, group the runs you want to compare in [Weights & Biases](https://wandb.ai) by assigning the **same tag** to each run (in the run’s overview or via the API). The tools below use that tag against the `minerva-models` project under your W&B entity (set `WANDB_ENTITY` and use `wandb login` as needed).

### Test evaluation (`eval`)

Generate the `python -m src.scripts.eval ...` commands for checkpoints that still need `test_results` (skipped if an `.npz` for that dataset already exists):

```bash
python -m src.scripts.print_eval_commands --wandb-flag <TAG>
```

`--wandb-flag` only lists runs whose checkpoint folder name matches a wandb run name with that tag; omit it to consider every run under `--ckpt-dir` (default: see `--help`). Run each printed line locally. **Evaluation is very small and fast**—it is fine to run on **login nodes** without a GPU job.

### Notebooks

1. Open `notebooks/Eval_Classification.ipynb` or `notebooks/Eval_Regression.ipynb`.
2. Set the `WANDB_TAG` variable at the top to match your tag (both notebooks use this to query runs).
3. Run all cells.

To run the same notebooks headlessly from the repo root (requires `nbconvert`; e.g. `pip install nbconvert`):

```bash
cd notebooks
# Execute (updates notebooks in place with fresh outputs)
jupyter nbconvert --to notebook --execute Eval_Regression.ipynb --inplace
jupyter nbconvert --to notebook --execute Eval_Classification.ipynb --inplace
# Export static copies (HTML; use --to pdf instead if pandoc/LaTeX are available, or --to webpdf with Chromium)
jupyter nbconvert --to html Eval_Regression.ipynb
jupyter nbconvert --to html Eval_Classification.ipynb
```

Classification evaluation covers tagging and related metrics; regression evaluation covers energy-scale and scaling plots. Figures and PDFs are written under paths configured in each notebook (typically under `out/`).

## 6) Event displays

```bash
python -m src.scripts.make_event_displays \
  --input_file <PATH_TO_ROOT_FILE> \
  --output_dir <PATH_TO_OUTPUT_DIR> \
  --n_events 10
```
