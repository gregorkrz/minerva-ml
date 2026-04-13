"""Integration tests: data loading, training loop, and evaluation on synthetic data."""

import os
import subprocess
import sys

import pytest


@pytest.fixture()
def _no_wandb(monkeypatch):
    """Ensure wandb is disabled for all tests in this module."""
    monkeypatch.setenv("WANDB_MODE", "disabled")


class TestDataLoading:

    def test_load_regression_dataset(self, synthetic_data_dir):
        from src.dataset.dataloader import load_data, Task

        task = Task(
            type="regression",
            regress_E_available_no_muon=True,
            class_label_idx=9,
            regress_log=False,
        )
        loader, class_weights = load_data(
            dataset_name="minerva_1A",
            path=synthetic_data_dir,
            batch=8,
            dataset_type="train",
            task=task,
            use_cond=True,
            use_pid=True,
            pid_idx=4,
            num_workers=0,
            distributed=False,
            shuffle=False,
            max_particles=33,
            use_energy_sums=True,
        )
        assert class_weights is None
        batch = next(iter(loader))
        assert batch["X"].shape[0] == 8
        assert batch["X"].shape[1] == 33
        assert batch["y"].dtype == torch.float32

    def test_load_classifier_dataset(self, synthetic_data_dir):
        from src.dataset.dataloader import load_data, Task

        task = Task(
            type="classifier",
            classification_event_type=True,
            class_idx=[1, 2, 3, 4, 8],
            class_idx_map={1: 0, 2: 1, 3: 2, 4: 3, 8: 4},
            class_label_idx=1,
        )
        loader, class_weights = load_data(
            dataset_name="minerva_1A",
            path=synthetic_data_dir,
            batch=8,
            dataset_type="train",
            task=task,
            use_cond=True,
            use_pid=True,
            pid_idx=4,
            num_workers=0,
            distributed=False,
            shuffle=False,
            max_particles=33,
        )
        assert class_weights is not None
        batch = next(iter(loader))
        assert batch["y"].dtype == torch.long


class TestTrainingCLI:
    """Run the training script as a subprocess to validate the full CLI pipeline."""

    def test_vit_regression(self, synthetic_data_dir, tmp_path, _no_wandb):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.scripts.train",
                "-bs",
                "8",
                "--mode",
                "regression",
                "-E-available-no-muon",
                "-name",
                "ci_vit_reg",
                "--d_model",
                "32",
                "--depth",
                "1",
                "--n_heads",
                "2",
                "--max_steps",
                "3",
                "--warmup_steps",
                "1",
                "--eval_interval",
                "3",
                "--log_interval",
                "3",
                "--save_interval",
                "3",
                "--no_wandb",
                "--num_workers",
                "1",
                "--data_path",
                synthetic_data_dir,
                "--output_dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        assert "Training complete!" in result.stdout

    def test_omnilearned_regression(self, synthetic_data_dir, tmp_path, _no_wandb):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.scripts.train",
                "-bs",
                "8",
                "--mode",
                "regression",
                "-E-available-no-muon",
                "-name",
                "ci_ol_reg",
                "--max_steps",
                "2",
                "--warmup_steps",
                "1",
                "--eval_interval",
                "2",
                "--log_interval",
                "2",
                "--save_interval",
                "2",
                "--no_wandb",
                "--num_workers",
                "1",
                "--data_path",
                synthetic_data_dir,
                "--output_dir",
                str(tmp_path),
                "--use-omnilearned",
                "small",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        assert "Training complete!" in result.stdout

    def test_vit_classifier(self, synthetic_data_dir, tmp_path, _no_wandb):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.scripts.train",
                "-bs",
                "8",
                "--mode",
                "classifier",
                "-npi2",
                "-name",
                "ci_vit_cls",
                "--d_model",
                "32",
                "--depth",
                "1",
                "--n_heads",
                "2",
                "--max_steps",
                "2",
                "--warmup_steps",
                "1",
                "--eval_interval",
                "2",
                "--log_interval",
                "2",
                "--save_interval",
                "2",
                "--no_wandb",
                "--num_workers",
                "1",
                "--data_path",
                synthetic_data_dir,
                "--output_dir",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        assert "Training complete!" in result.stdout

    def test_condonly_regression(self, synthetic_data_dir, tmp_path, _no_wandb):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.scripts.train",
                "-bs",
                "8",
                "--mode",
                "regression",
                "-E-available-no-muon",
                "-name",
                "ci_cond_reg",
                "--d_model",
                "32",
                "--mlp_layers",
                "2",
                "--max_steps",
                "2",
                "--warmup_steps",
                "1",
                "--eval_interval",
                "2",
                "--log_interval",
                "2",
                "--save_interval",
                "2",
                "--no_wandb",
                "--num_workers",
                "1",
                "--data_path",
                synthetic_data_dir,
                "--output_dir",
                str(tmp_path),
                "--cond_only",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        assert "Training complete!" in result.stdout

    def test_bert_regression(self, synthetic_data_dir, tmp_path, _no_wandb):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.scripts.train",
                "-bs",
                "8",
                "--mode",
                "regression",
                "-E-available-no-muon",
                "-name",
                "ci_bert_reg",
                "--max_steps",
                "2",
                "--warmup_steps",
                "1",
                "--eval_interval",
                "2",
                "--log_interval",
                "2",
                "--save_interval",
                "2",
                "--no_wandb",
                "--num_workers",
                "1",
                "--data_path",
                synthetic_data_dir,
                "--output_dir",
                str(tmp_path),
                "--use-bert",
                "tiny",
                "--bert-random-init",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0, f"Training failed:\n{result.stderr}"
        assert "Training complete!" in result.stdout


# Allow the dataloader test to reference torch without an explicit import at the top,
# keeping the import inside the test module scope alongside the src imports.
import torch  # noqa: E402
