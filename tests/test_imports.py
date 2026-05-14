"""Verify that all project dependencies and internal modules are importable."""

import pytest


class TestCoreDependencies:
    """Third-party packages required by the project."""

    def test_torch(self):
        import torch

        assert torch.__version__

    def test_numpy(self):
        import numpy

        assert numpy.__version__

    def test_wandb(self):
        import wandb

        assert wandb.__version__

    def test_transformers(self):
        import transformers

        assert transformers.__version__

    def test_pytorch_optimizer(self):
        from pytorch_optimizer import Lion

        assert Lion is not None

    def test_einops(self):
        import einops

        assert einops.__version__

    def test_sklearn(self):
        import sklearn

        assert sklearn.__version__

    def test_h5py(self):
        import h5py

        assert h5py.__version__

    def test_matplotlib(self):
        import matplotlib

        assert matplotlib.__version__

    def test_plotly(self):
        import plotly

        assert plotly.__version__

    def test_tqdm(self):
        import tqdm

        assert tqdm.__version__

    def test_uproot(self):
        import uproot

        assert uproot.__version__

    def test_awkward(self):
        import awkward

        assert awkward.__version__


class TestInternalModules:
    """Project-internal module imports."""

    def test_vit(self):
        from src.models.vit import PointGlobalMixedViT, PointGlobalMixedViTConfig

        assert PointGlobalMixedViT is not None

    def test_omnilearned(self):
        from src.models.omnilearned import PET2, get_model_parameters

        assert PET2 is not None

    def test_hyperscale(self):
        from src.models.hyperscale import (
            ParticleVIT,
            ParticleVIT_Embedding,
            ParticleVIT_Pool,
        )

        assert ParticleVIT is not None
        assert ParticleVIT_Embedding is not None
        assert ParticleVIT_Pool is not None

    def test_dataloader(self):
        from src.dataset.dataloader import load_data, Task

        assert Task is not None

    def test_constants(self):
        from src.constants.dataset import GLOBAL_COND_BASE_DIM

        assert GLOBAL_COND_BASE_DIM == 10

    def test_physics_constants(self):
        from src.constants.physics import pdg_masses

        assert pdg_masses[2212] > 0

    def test_preprocessing(self):
        from src.dataset.preprocessing import get_dense

        assert get_dense is not None

    def test_utils(self):
        from src.utils.utils import fetch_runs_from_wandb

        assert fetch_runs_from_wandb is not None

    def test_train_module(self):
        from src.scripts.train import (
            set_seed,
            prepare_batch,
            prepare_batch_omnilearned,
            create_task,
            CondOnlyMLP,
        )

        assert set_seed is not None
