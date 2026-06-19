"""Wandb run-name → model-key parsing (no API calls)."""

from __future__ import annotations

from src.utils.utils import parse_binned_classifier_model_cap, parse_hyperscale_model_cap


def test_hyperscale_pretrained_regression():
    name = (
        "Run_0601_HyperScale_small_embedding_regression_-1_seed55_20260601_120000"
    )
    assert parse_hyperscale_model_cap(name, task="regression") == (
        "HyperScale-small",
        -1,
    )
    assert parse_hyperscale_model_cap(name, task="classifier") is None


def test_hyperscale_rw_classifier():
    name = "Run_0601_HyperScale_medium_rw_embedding_classifier_-1_seed56"
    assert parse_hyperscale_model_cap(name, task="classifier") == (
        "HyperScale-medium-rw",
        -1,
    )
    assert parse_hyperscale_model_cap(name, task="regression") is None


def test_hyperscale_rw_before_pretrained():
    rw = "Run_0601_HyperScale_small_rw_embedding_regression_-1_seed55"
    pt = "Run_0601_HyperScale_small_embedding_regression_-1_seed55"
    assert parse_hyperscale_model_cap(rw, task="regression") == (
        "HyperScale-small-rw",
        -1,
    )
    assert parse_hyperscale_model_cap(pt, task="regression") == ("HyperScale-small", -1)


def test_binned_classifier_ccn1pipm_maps_to_weigh1():
    name = (
        "Run_1703_classifier_Transformer1_data_cap_-1_seed_55_"
        "binnedW_CCN1pipm_20260609_213113"
    )
    assert parse_binned_classifier_model_cap(name) == (
        "Transformer-xsmall-Weigh1",
        -1,
    )


def test_binned_classifier_ccn1pipm_bin_maps_to_weigh2():
    name = (
        "Run_1703_classifier_Transformer1_data_cap_-1_seed_55_"
        "binnedW_CCN1pipmBin_20260611_120000"
    )
    assert parse_binned_classifier_model_cap(name) == (
        "Transformer-xsmall-Weigh2",
        -1,
    )


def test_binned_classifier_plain_transformer1_not_matched():
    name = "Run_1703_classifier_Transformer1_data_cap_-1_seed_55_20260609_213113"
    assert parse_binned_classifier_model_cap(name) is None


def test_non_hyperscale_returns_none():
    assert (
        parse_hyperscale_model_cap(
            "Run_0601_BERT_tiny_regression_-1_seed55", task="regression"
        )
        is None
    )
