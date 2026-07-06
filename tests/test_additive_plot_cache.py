"""Tests for additive plot-cache merge helpers."""

from __future__ import annotations

import pytest

from src.eval._cache_additive import (
    light_cache_model_keys,
    merge_light_classification_specs,
    merge_model_metrics_tree,
    merge_prc_tree,
    metrics_cache_model_keys,
    resolve_models_to_add,
)


def test_resolve_models_to_add_auto():
    assert resolve_models_to_add(
        {"A", "B", "C"},
        {"A", "B"},
        None,
    ) == ["C"]


def test_resolve_models_to_add_explicit():
    assert resolve_models_to_add(
        {"A", "B", "C"},
        {"A"},
        ["B", "C"],
    ) == ["B", "C"]


def test_resolve_models_to_add_unknown_raises():
    with pytest.raises(ValueError, match="not in source pickle"):
        resolve_models_to_add({"A"}, set(), ["B"])


def test_merge_model_metrics_tree():
    existing = {
        "cc1pi": {
            "1A": {
                "all": {"MLP": {"auprc": [1.0]}},
            }
        }
    }
    partial = {
        "cc1pi": {
            "1A": {
                "all": {"BDT-binnedW": {"auprc": [0.9]}},
                2: {"BDT-binnedW": {"auprc": [0.8]}},
            }
        }
    }
    merge_model_metrics_tree(existing, partial)
    assert existing["cc1pi"]["1A"]["all"]["MLP"]["auprc"] == [1.0]
    assert existing["cc1pi"]["1A"]["all"]["BDT-binnedW"]["auprc"] == [0.9]
    assert existing["cc1pi"]["1A"][2]["BDT-binnedW"]["auprc"] == [0.8]


def test_merge_prc_tree():
    existing = {"cc1pi": {"1A": {"MLP": {"auprc_mean": 0.5}}}}
    partial = {"cc1pi": {"1A": {"BDT-binnedW": {"auprc_mean": 0.4}}}}
    merge_prc_tree(existing, partial)
    assert set(existing["cc1pi"]["1A"]) == {"MLP", "BDT-binnedW"}


def test_merge_light_classification_specs():
    existing = [
        {
            "type": "1x3",
            "filename": "eval_classification_light_cc1pi_q3_1A.pdf",
            "all_metrics": {"MLP": {"auprc": [1.0]}},
        }
    ]
    new = [
        {
            "type": "1x3",
            "filename": "eval_classification_light_cc1pi_q3_1A.pdf",
            "all_metrics": {"BDT-binnedW": {"auprc": [0.9]}},
        },
        {
            "type": "1x3",
            "filename": "eval_classification_light_cc1pi_W_1A.pdf",
            "all_metrics": {"BDT-binnedW": {"auprc": [0.8]}},
        },
    ]
    merged = merge_light_classification_specs(existing, new)
    by_fn = {s["filename"]: s for s in merged}
    assert set(by_fn) == {
        "eval_classification_light_cc1pi_q3_1A.pdf",
        "eval_classification_light_cc1pi_W_1A.pdf",
    }
    assert set(by_fn["eval_classification_light_cc1pi_q3_1A.pdf"]["all_metrics"]) == {
        "MLP",
        "BDT-binnedW",
    }


def test_cache_model_key_helpers():
    metrics = {"confusion_matrices": {"MLP": {}, "BDT": {}}}
    assert metrics_cache_model_keys(metrics) == {"MLP", "BDT"}

    light = {
        "specs": [
            {"all_metrics": {"MLP": {}, "BERT-tiny": {}}},
            {"all_metrics": {"BDT-binnedW": {}}},
        ]
    }
    assert light_cache_model_keys(light) == {"MLP", "BERT-tiny", "BDT-binnedW"}
