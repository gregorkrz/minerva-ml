# AGENTS.md

## Cursor Cloud specific instructions

This is a Python ML research codebase for neutrino event classification and energy regression on MINERvA particle physics data. It is **not** a web application or a multi-service project — it is a single-service Python package run directly from the repo root.

### Running scripts

All scripts are invoked as Python modules from the repo root. On NERSC, use the **`omni`** conda environment (PyTorch, h5py, matplotlib, etc.):

```bash
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.scripts.train ...
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.scripts.eval ...
/global/homes/g/gregork/.conda/envs/omni/bin/python -m src.scripts.split_dataset ...
```

See `README.md` for full command examples.

### Key caveats

- **No GPU in Cloud Agent VMs.** Training runs on CPU; use small batch sizes (`-bs 10`) and few steps (`--max_steps 5`) for validation. Always pass `--no_wandb` to avoid W&B login prompts.
- **num_workers must be >= 1** when `distributed=True` (the default), because the DataLoader sets `prefetch_factor=4` which requires multiprocessing. Use `--num_workers 1` in Cloud Agent environments.
- **No requirements.txt or pyproject.toml.** Dependencies are inferred from imports. The update script installs them via pip.
- **Dataset files use `torch.nested.nested_tensor` with `layout=torch.jagged`.** When creating synthetic test data, use `torch.nested.nested_tensor(tensors, layout=torch.jagged)` so `.offsets()` and `.values()` methods work. Truth labels and global features should be `torch.Tensor` (not numpy) for `weights_only=True` compatibility.
- **No linter configuration** is checked in. The codebase has pre-existing `ruff` warnings (unused imports, f-string placeholders) that are not enforced.
- **W&B (`wandb`) and real data** are needed for the full workflow (experiment tracking, evaluation notebooks). For basic dev/test, use `--no_wandb` and synthetic data.
- **The `preprocess_dataset.py` script requires ROOT (PyROOT)**, which is not available in Cloud Agent VMs. Preprocessing of raw ROOT files is not possible here; use the HuggingFace preprocessed dataset or synthetic data instead.


On perlmutter, use the /global/homes/g/gregork/.conda/envs/omni/bin/python interpreter.
