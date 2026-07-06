"""BDT validation loss must aggregate like train.evaluate (batched mean × batch size)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.scripts.train import (
    _bdt_classification_batch_loss,
    _bdt_regression_batch_loss,
    evaluate_bdt_val_loss,
)


class _MockClf:
    classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        n = X.shape[0]
        p = np.array([[0.7, 0.2, 0.1]] * n, dtype=np.float64)
        return p


class _MockReg:
    def predict(self, X):
        return np.linspace(0.1, 0.9, X.shape[0], dtype=np.float64)


class _FakeLoader:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _classifier_task():
    from src.dataset.dataloader import Task

    return Task(
        type="classifier",
        classification_event_type=True,
        class_idx=[1, 2, 3, 4, 8],
        class_idx_map={1: 0, 2: 1, 3: 2, 4: 3, 8: 4},
        class_label_idx=1,
        class_weights=[1.0, 2.0, 3.0, 4.0, 5.0],
    )


def _regression_args():
    return SimpleNamespace(
        mode="regression",
        use_pid=True,
        coord_dim=2,
        pid_idx=4,
        include_E_sum=True,
        zero_cond_feature=None,
        log_MSE_loss=False,
        log1p_loss=False,
        weighted_regression_loss=False,
    )


def _classifier_args():
    return SimpleNamespace(
        mode="classifier",
        use_pid=True,
        coord_dim=2,
        pid_idx=4,
        include_E_sum=True,
        zero_cond_feature=None,
    )


def _make_cond_batch(labels, *, feat_dim=16):
    y = torch.tensor(labels)
    b = y.shape[0]
    batch = {
        "X": torch.zeros(b, 4, 9, dtype=torch.float32),
        "y": y,
        "attention_mask": torch.ones(b, 4, dtype=torch.float32),
        "cond": torch.randn(b, 10, dtype=torch.float32),
        "energy_sums": torch.ones(b, 6, dtype=torch.float32),
    }
    return batch


def test_bdt_classification_batched_matches_eval_aggregate():
    """Batched CE aggregate equals sum(mean_b * n_b) / N like evaluate()."""
    task = _classifier_task()
    args = _classifier_args()
    clf = _MockClf()
    num_classes = len(task.class_idx)
    class_weights = torch.tensor(task.class_weights, dtype=torch.float32)

    batches = [
        _make_cond_batch([0, 1, 2]),
        _make_cond_batch([1, 2]),
    ]
    loader = _FakeLoader(batches)

    batched = evaluate_bdt_val_loss(clf, loader, args, task, use_binned_loss=False)

    total_loss = 0.0
    total_samples = 0
    for batch in batches:
        from src.scripts.train import _bdt_cond_batch

        X, y, _ = _bdt_cond_batch(batch, args)
        loss_b = _bdt_classification_batch_loss(
            clf, X, y, num_classes, task, False, class_weights=class_weights,
        )
        n = len(y)
        total_loss += loss_b * n
        total_samples += n
    manual = total_loss / total_samples
    assert batched == pytest.approx(manual)


def test_bdt_regression_batched_matches_eval_aggregate():
    task = SimpleNamespace(type="regression")
    args = _regression_args()
    reg = _MockReg()
    batches = [
        _make_cond_batch([0.5, 1.0, 2.0], feat_dim=16),
        _make_cond_batch([3.0], feat_dim=16),
    ]
    for batch in batches:
        batch["y"] = batch["y"].float()
    loader = _FakeLoader(batches)

    batched = evaluate_bdt_val_loss(reg, loader, args, task)

    total_loss = 0.0
    total_samples = 0
    for batch in batches:
        from src.scripts.train import _bdt_cond_batch

        X, y, _ = _bdt_cond_batch(batch, args)
        pred = reg.predict(X)
        loss_b = _bdt_regression_batch_loss(pred, y, args)
        n = len(y)
        total_loss += loss_b * n
        total_samples += n
    assert batched == pytest.approx(total_loss / total_samples)


def test_bdt_weighted_classification_batch_matches_cross_entropy():
    task = _classifier_task()
    clf = _MockClf()
    X = np.random.randn(5, 16).astype(np.float32)
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    loss = _bdt_classification_batch_loss(
        clf, X, y, 5, task, True, sample_weights=weights,
    )
    proba = clf.predict_proba(X)
    logits = torch.log(torch.from_numpy(proba).clamp(min=1e-12)).float()
    expected = F.cross_entropy(
        logits, torch.from_numpy(y).long(), reduction="none",
    )
    expected = (expected * torch.from_numpy(weights)).mean().item()
    assert loss == pytest.approx(expected)
