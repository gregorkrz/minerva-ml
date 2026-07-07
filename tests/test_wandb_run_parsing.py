"""Wandb run-name → model-key parsing (no API calls)."""

from __future__ import annotations

from src.utils.utils import (
    classification_model_cap_from_name,
    parse_binned_classifier_model_cap,
    parse_binned_mlp_model_cap,
    parse_hyperscale_model_cap,
    parse_predict_baseline_binned_mlp_model_cap,
    parse_predict_baseline_classifier_model_cap,
    regression_model_cap_from_name,
)


def test_hyperscale_pretrained_regression():
    name = "Run_0601_HyperScale_small_embedding_regression_-1_seed55_20260601_120000"
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


def test_predict_baseline_transformer_small():
    name = (
        "Run_1703_classifier_Transformer3NR_data_cap_-1_seed_55_"
        "predictBaseline_20260616_150730"
    )
    assert parse_predict_baseline_classifier_model_cap(name) == (
        "Transformer-small-Baseline",
        -1,
    )


def test_predict_baseline_mlp():
    name = (
        "Run_cond_only_lowLR_MLP3_classifier_predictBaseline_NR_full_seed55_"
        "20260616_150730"
    )
    assert parse_predict_baseline_classifier_model_cap(name) == ("MLP-Baseline", -1)


def test_predict_baseline_mlp_binned_w_maps_to_plot_key():
    name = (
        "Run_cond_only_lowLR_MLP3_classifier_predictBaseline_binnedW_CCN1pipmBin_"
        "NR_full_seed55_20260625_110754"
    )
    assert parse_predict_baseline_binned_mlp_model_cap(name) == (
        "MLP-predictBaseline-binnedW",
        -1,
    )
    assert parse_predict_baseline_classifier_model_cap(name) == (
        "MLP-predictBaseline-binnedW",
        -1,
    )


def test_binned_mlp_ccn1pipm_maps_to_mlp_binned_w():
    name = (
        "Run_cond_only_lowLR_MLP3_classifier_binnedW_CCN1pipm_NR_full_seed55_"
        "20260625_120000"
    )
    assert parse_binned_mlp_model_cap(name) == ("MLP-binnedW", -1)


def test_binned_mlp_ccn1pipm_bin_maps_to_mlp_binned_w_bin():
    name = (
        "Run_cond_only_lowLR_MLP3_classifier_binnedW_CCN1pipmBin_NR_full_seed55_"
        "20260625_120000"
    )
    assert parse_binned_mlp_model_cap(name) == ("MLP-binnedW-Bin", -1)


def test_binned_mlp_does_not_match_predict_baseline_binned():
    name = (
        "Run_cond_only_lowLR_MLP3_classifier_predictBaseline_binnedW_CCN1pipmBin_"
        "NR_full_seed55_20260625_110754"
    )
    assert parse_binned_mlp_model_cap(name) is None


def test_binned_mlp_does_not_match_plain_mlp():
    name = "Run_cond_only_lowLR_MLP3_classifier_NR_full_seed55_20260625_120000"
    assert parse_binned_mlp_model_cap(name) is None


def test_predict_baseline_mlp_binned_w_does_not_match_plain_baseline():
    name = (
        "Run_cond_only_lowLR_MLP3_classifier_predictBaseline_binnedW_CCN1pipmBin_"
        "NR_full_seed55_20260625_110754"
    )
    assert parse_predict_baseline_classifier_model_cap(name) != ("MLP-Baseline", -1)


def test_predict_baseline_plain_transformer_small_not_matched():
    name = "Run_1703_classifier_Transformer3NR_data_cap_-1_seed_55_20260616_150730"
    assert parse_predict_baseline_classifier_model_cap(name) is None


def test_bdt_mc_truth_binned_w_maps_to_bdt_binned_w():
    name = "Run_BDT_classifier_binnedW_CCN1pipm_seed55_20260630_120000"
    assert classification_model_cap_from_name(name) == ("BDT-binnedW", -1)


def test_bdt_mc_truth_plain_maps_to_bdt():
    name = "Run_BDT_classifier_seed55_20260630_120000"
    assert classification_model_cap_from_name(name) == ("BDT", -1)


def test_bdt_regression_maps_to_bdt():
    name = "Run_BDT_regression_NR_full_seed55_20260630_120000"
    assert regression_model_cap_from_name(name) == ("BDT", -1)


def test_bdt_predict_baseline_maps_to_bdt_bc():
    name = "Run_BDT_classifier_predictBaseline_seed55_20260630_120000"
    assert classification_model_cap_from_name(name) == ("BDT-BC", -1)


def test_bdt_predict_baseline_binned_w_maps_to_bdt_bc_binned_w():
    name = (
        "Run_BDT_classifier_predictBaseline_binnedW_CCN1pipmBin_seed55_"
        "20260630_120000"
    )
    assert classification_model_cap_from_name(name) == ("BDT-BC-binnedW", -1)


def test_non_hyperscale_returns_none():
    assert (
        parse_hyperscale_model_cap(
            "Run_0601_BERT_tiny_regression_-1_seed55", task="regression"
        )
        is None
    )
