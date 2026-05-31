"""Test model instantiation and forward passes with dummy data."""

import pytest
import torch
import torch.nn as nn

from src.models.vit import PointGlobalMixedViT, PointGlobalMixedViTConfig
from src.models.omnilearned import PET2, get_model_parameters
from src.constants.dataset import GLOBAL_COND_BASE_DIM

B, N = 4, 33  # batch size, max particles


class TestPointGlobalMixedViT:

    @pytest.fixture()
    def model(self):
        cfg = PointGlobalMixedViTConfig(
            point_cont_dim=9,
            point_cat_num_classes=[8],
            global_cont_dim=GLOBAL_COND_BASE_DIM + 6,
            global_cat_num_classes=[],
            coord_dim=2,
            d_model=64,
            depth=2,
            n_heads=4,
            mlp_ratio=4.0,
            dropout=0.0,
            attn_dropout=0.0,
            use_cls_token=True,
            use_event_token=True,
            cat_emb_dim=16,
        )
        m = PointGlobalMixedViT(cfg)
        m.head = nn.Linear(64, 1)
        return m

    def test_forward_shape(self, model):
        point_cont = torch.randn(B, N, 9)
        point_cats = [torch.randint(0, 8, (B, N))]
        pos = torch.randn(B, N, 2)
        global_cont = torch.randn(B, GLOBAL_COND_BASE_DIM + 6)
        num_special = 2
        attn_mask = torch.ones(B, N + num_special)
        attn_mask = (attn_mask > 0).unsqueeze(1).unsqueeze(2)

        model.eval()
        with torch.no_grad():
            features = model(
                point_cont=point_cont,
                point_cats=point_cats,
                pos=pos,
                global_cont=global_cont,
                global_cats=None,
                attn_mask=attn_mask,
            )
            output = model.head(features)

        assert output.shape == (B, 1)

    def test_no_event_token(self):
        cfg = PointGlobalMixedViTConfig(
            point_cont_dim=9,
            point_cat_num_classes=[8],
            global_cont_dim=0,
            global_cat_num_classes=[],
            coord_dim=2,
            d_model=64,
            depth=2,
            n_heads=4,
            use_cls_token=True,
            use_event_token=False,
            cat_emb_dim=16,
        )
        m = PointGlobalMixedViT(cfg)
        m.head = nn.Linear(64, 1)

        attn_mask = torch.ones(B, N + 1)  # only CLS token
        attn_mask = (attn_mask > 0).unsqueeze(1).unsqueeze(2)

        m.eval()
        with torch.no_grad():
            features = m(
                point_cont=torch.randn(B, N, 9),
                point_cats=[torch.randint(0, 8, (B, N))],
                pos=torch.randn(B, N, 2),
                global_cont=None,
                global_cats=None,
                attn_mask=attn_mask,
            )
            out = m.head(features)
        assert out.shape == (B, 1)


class TestOmniLearnedPET2:

    @pytest.fixture()
    def model(self):
        params = get_model_parameters("small")
        return PET2(
            input_dim=4,
            use_int=False,
            local_int=False,
            int_type="lhc",
            conditional=True,
            cond_dim=16,
            pid=True,
            pid_dim=8,
            add_info=True,
            add_dim=5,
            mode="regression",
            num_classes=1,
            num_gen_classes=1,
            mlp_drop=0.0,
            attn_drop=0.0,
            feature_drop=0.0,
            num_coord=2,
            K=10,
            **params,
        )

    def test_forward_regression(self, model):
        X = torch.randn(B, N, 4)
        y = torch.randn(B)
        cond = torch.randn(B, 16)
        pid = torch.randint(0, 8, (B, N))
        add_info = torch.randn(B, N, 5)

        model.eval()
        with torch.no_grad():
            outputs = model(X, y, cond=cond, pid=pid, add_info=add_info)

        assert outputs["y_pred"].shape == (B, 1)

    def test_forward_classifier(self):
        params = get_model_parameters("small")
        model = PET2(
            input_dim=4,
            use_int=False,
            local_int=False,
            int_type="lhc",
            conditional=True,
            cond_dim=16,
            pid=True,
            pid_dim=8,
            add_info=True,
            add_dim=5,
            mode="classifier",
            num_classes=5,
            num_gen_classes=1,
            mlp_drop=0.0,
            attn_drop=0.0,
            feature_drop=0.0,
            num_coord=2,
            K=10,
            **params,
        )
        model.eval()
        with torch.no_grad():
            outputs = model(
                torch.randn(B, N, 4),
                torch.zeros(B),
                cond=torch.randn(B, 16),
                pid=torch.randint(0, 8, (B, N)),
                add_info=torch.randn(B, N, 5),
            )
        assert outputs["y_pred"].shape == (B, 5)


class TestCondOnlyMLP:

    def test_forward_regression(self):
        from src.scripts.train import CondOnlyMLP

        model = CondOnlyMLP(
            input_dim=16,
            hidden_dim=64,
            output_dim=1,
            n_layers=2,
            dropout=0.0,
            output_positive=True,
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(B, 16))
        assert out.shape == (B, 1)
        assert (
            out >= 0
        ).all(), "output_positive=True should produce non-negative values"

    def test_forward_classifier(self):
        from src.scripts.train import CondOnlyMLP

        model = CondOnlyMLP(
            input_dim=16,
            hidden_dim=64,
            output_dim=5,
            n_layers=2,
            dropout=0.0,
            output_positive=False,
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(B, 16))
        assert out.shape == (B, 5)


class TestHyperScale:
    """Vendored HyperScale ParticleViT models (no wrapper)."""

    def test_basic_forward(self):
        from src.models.hyperscale import ParticleVIT

        m = ParticleVIT(
            num_features=10,
            num_classes=1,
            embed_dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=8 / 3,
        )
        m.eval()
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, 20:] = False
        with torch.no_grad():
            out = m(torch.randn(B, N, 10), attn_mask=mask)
        assert out.shape == (B, 1)

    def test_embedding_forward(self):
        from src.models.hyperscale import ParticleVIT_Embedding

        m = ParticleVIT_Embedding(
            num_features=9,
            num_classes=5,
            embed_dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=8 / 3,
        )
        m.eval()
        X = torch.randn(B, N, 9)
        X[:, :, 4] = torch.randint(0, 9, (B, N)).float()
        mask = torch.ones(B, N, dtype=torch.bool)
        with torch.no_grad():
            out = m(X, attn_mask=mask)
        assert out.shape == (B, 5)

    def test_embedding_requires_9_features(self):
        from src.models.hyperscale import ParticleVIT_Embedding

        with pytest.raises(ValueError, match="expects 9 features"):
            ParticleVIT_Embedding(
                num_features=10,
                num_classes=1,
                embed_dim=64,
                depth=1,
                num_heads=4,
                mlp_ratio=8 / 3,
            )

    def test_pool_forward(self):
        from src.models.hyperscale import ParticleVIT_Pool

        m = ParticleVIT_Pool(
            num_features=10,
            num_classes=1,
            embed_dim=64,
            depth=2,
            num_heads=4,
            mlp_ratio=8 / 3,
        )
        m.eval()
        mask = torch.ones(B, N, dtype=torch.bool)
        mask[:, 25:] = False
        with torch.no_grad():
            out = m(torch.randn(B, N, 10), attn_mask=mask)
        assert out.shape == (B, 1)


class TestHyperScaleBaseline:
    """Train.py wrapper that adds an optional projected global token."""

    def test_basic_forward(self):
        from src.scripts.train import HyperScaleBaseline

        model = HyperScaleBaseline(
            input_dim=10,
            output_dim=1,
            embed_dim=64,
            depth=2,
            num_heads=4,
            variant="basic",
            mlp_ratio=8 / 3,
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(B, N, 10), torch.ones(B, N))
        assert out.shape == (B, 1)

    def test_basic_with_global_token(self):
        from src.scripts.train import HyperScaleBaseline

        model = HyperScaleBaseline(
            input_dim=10,
            output_dim=5,
            embed_dim=64,
            depth=2,
            num_heads=4,
            variant="basic",
            mlp_ratio=8 / 3,
            global_cont_dim=16,
        )
        model.eval()
        with torch.no_grad():
            out = model(
                torch.randn(B, N, 10),
                torch.ones(B, N),
                global_cont=torch.randn(B, 16),
            )
        assert out.shape == (B, 5)

    def test_global_token_requires_cond(self):
        from src.scripts.train import HyperScaleBaseline

        model = HyperScaleBaseline(
            input_dim=10,
            output_dim=1,
            embed_dim=64,
            depth=2,
            num_heads=4,
            variant="basic",
            mlp_ratio=8 / 3,
            global_cont_dim=16,
        )
        model.eval()
        with pytest.raises(ValueError, match="global_cont is missing"):
            with torch.no_grad():
                _ = model(torch.randn(B, N, 10), torch.ones(B, N))

    def test_embedding_variant(self):
        from src.scripts.train import HyperScaleBaseline

        model = HyperScaleBaseline(
            input_dim=9,
            output_dim=1,
            embed_dim=64,
            depth=2,
            num_heads=4,
            variant="embedding",
            mlp_ratio=8 / 3,
        )
        model.eval()
        X = torch.randn(B, N, 9)
        X[:, :, 4] = torch.randint(0, 9, (B, N)).float()
        with torch.no_grad():
            out = model(X, torch.ones(B, N))
        assert out.shape == (B, 1)

    def test_pool_variant(self):
        from src.scripts.train import HyperScaleBaseline

        model = HyperScaleBaseline(
            input_dim=10,
            output_dim=3,
            embed_dim=64,
            depth=2,
            num_heads=4,
            variant="pool",
            mlp_ratio=8 / 3,
            global_cont_dim=16,
        )
        model.eval()
        with torch.no_grad():
            out = model(
                torch.randn(B, N, 10),
                torch.ones(B, N),
                global_cont=torch.randn(B, 16),
            )
        assert out.shape == (B, 3)

    def test_unknown_variant_rejected(self):
        from src.scripts.train import HyperScaleBaseline

        with pytest.raises(ValueError, match="Unknown HyperScale variant"):
            HyperScaleBaseline(
                input_dim=10,
                output_dim=1,
                embed_dim=64,
                depth=2,
                num_heads=4,
                variant="bogus",
                mlp_ratio=8 / 3,
            )

    def test_load_pretrained_transfers_encoder_skips_head(self, tmp_path):
        """Pretrained loader copies token_embed/blocks/cls_token weights but
        leaves the task-specific head and the wrapper-only global_proj at init."""
        from src.scripts.train import HyperScaleBaseline
        from src.models.hyperscale import load_pretrained_hyperscale

        # Source: trained on 1-class regression head, no global token.
        src = HyperScaleBaseline(
            input_dim=10,
            output_dim=1,
            embed_dim=64,
            depth=2,
            num_heads=4,
            variant="basic",
            mlp_ratio=8 / 3,
        )
        ckpt_path = tmp_path / "hs_source.pt"
        torch.save({"model_state_dict": src.state_dict()}, ckpt_path)

        # Target: same encoder shape but a 5-way classifier head and a new
        # global token projection.
        dst = HyperScaleBaseline(
            input_dim=10,
            output_dim=5,
            embed_dim=64,
            depth=2,
            num_heads=4,
            variant="basic",
            mlp_ratio=8 / 3,
            global_cont_dim=16,
        )

        head_before = dst.head.weight.detach().clone()
        gproj_before = dst.global_proj.weight.detach().clone()
        embed_before = dst.token_embed.weight.detach().clone()

        load_pretrained_hyperscale(dst, str(ckpt_path), verbose=False)

        # Encoder transferred.
        assert torch.allclose(dst.token_embed.weight, src.token_embed.weight)
        assert torch.allclose(dst.cls_token, src.cls_token)
        assert torch.allclose(
            dst.blocks[0].pre_attn_norm.weight, src.blocks[0].pre_attn_norm.weight
        )
        # token_embed actually moved away from its init.
        assert not torch.allclose(dst.token_embed.weight, embed_before)
        # Task head and global_proj are left at init (shape mismatch / not in ckpt).
        assert torch.allclose(dst.head.weight, head_before)
        assert torch.allclose(dst.global_proj.weight, gproj_before)


class TestHyperScaleAutofill:
    """--hs-pretrained alone should auto-fill arch from the checkpoint's saved args."""

    def test_autofill_overrides_missing_arch(self, tmp_path):
        import argparse
        from src.scripts.train import _maybe_autofill_hyperscale_args

        # Save a checkpoint that mimics src.scripts.train.save_checkpoint output:
        # {"args": vars(args), "model_state_dict": ...}.
        saved_args = {
            "use_hyperscale": "pool",
            "d_model": 128,
            "depth": 3,
            "n_heads": 4,
            "hs_mlp_ratio": 2.0,
        }
        ckpt = tmp_path / "fake_hs.pt"
        torch.save({"args": saved_args, "model_state_dict": {}}, ckpt)

        # User passes only --hs-pretrained (no --use-hyperscale, no arch flags).
        args = argparse.Namespace(
            hs_pretrained=str(ckpt),
            use_hyperscale=None,
            d_model=999,
            depth=999,
            n_heads=999,
            hs_mlp_ratio=999.0,
        )
        changed = _maybe_autofill_hyperscale_args(args)
        assert changed is True
        assert args.use_hyperscale == "pool"
        assert args.d_model == 128
        assert args.depth == 3
        assert args.n_heads == 4
        assert args.hs_mlp_ratio == 2.0

    def test_cli_variant_blocks_autofill(self, tmp_path):
        import argparse
        from src.scripts.train import _maybe_autofill_hyperscale_args

        saved_args = {"use_hyperscale": "pool", "d_model": 128}
        ckpt = tmp_path / "fake_hs.pt"
        torch.save({"args": saved_args}, ckpt)

        # User explicitly passed --use-hyperscale, so we trust their CLI.
        args = argparse.Namespace(
            hs_pretrained=str(ckpt),
            use_hyperscale="basic",
            d_model=64,
            depth=2,
            n_heads=4,
            hs_mlp_ratio=8 / 3,
        )
        changed = _maybe_autofill_hyperscale_args(args)
        assert changed is False
        assert args.use_hyperscale == "basic"
        assert args.d_model == 64

    def test_autofill_from_upstream_train_config_yaml(self, tmp_path):
        """When ckpt has no saved args, fall back to train_config.yaml next to it."""
        import argparse
        from src.scripts.train import _maybe_autofill_hyperscale_args

        # Upstream HyperScale dumps train_config.yaml alongside best_model.pt.
        # Use the exact format from gregorkrz/HyperScale's train.py.
        (tmp_path / "train_config.yaml").write_text(
            "model_type: ParticleVIT_Embedding\n"
            "model_params:\n"
            "  num_features: 9\n"
            "  num_classes: 210\n"
            "  embed_dim: 448\n"
            "  depth: 4\n"
            "  num_heads: 7\n"
            "  mlp_ratio: 2.6666666666666665\n"
        )
        ckpt = tmp_path / "best_model.pt"
        torch.save({"some_weight": torch.zeros(1)}, ckpt)  # raw, no "args" key

        args = argparse.Namespace(
            hs_pretrained=str(ckpt),
            use_hyperscale=None,
            d_model=0,
            depth=0,
            n_heads=0,
            hs_mlp_ratio=0.0,
        )
        changed = _maybe_autofill_hyperscale_args(args)
        assert changed is True
        assert args.use_hyperscale == "embedding"
        assert args.d_model == 448
        assert args.depth == 4
        assert args.n_heads == 7
        assert args.hs_mlp_ratio == pytest.approx(8 / 3)

    def test_autofill_needs_saved_args(self, tmp_path):
        import argparse
        import pytest as _pytest
        from src.scripts.train import _maybe_autofill_hyperscale_args

        # Bare state-dict checkpoint with no "args" key.
        ckpt = tmp_path / "raw_hs.pt"
        torch.save({"some_weight": torch.zeros(1)}, ckpt)
        args = argparse.Namespace(
            hs_pretrained=str(ckpt),
            use_hyperscale=None,
            d_model=0,
            depth=0,
            n_heads=0,
            hs_mlp_ratio=0.0,
        )
        with _pytest.raises(ValueError, match="no saved args"):
            _maybe_autofill_hyperscale_args(args)


class TestBertBaseline:

    def test_forward(self):
        from src.scripts.train import BertBaseline

        model = BertBaseline(
            input_dim=4,
            output_dim=1,
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            use_cls_token=False,
            bert_random_init=True,
        )
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(B, N, 4), torch.ones(B, N))
        assert out.shape == (B, 1)

    def test_forward_with_global_token(self):
        from src.scripts.train import BertBaseline

        model = BertBaseline(
            input_dim=4,
            output_dim=1,
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            use_cls_token=False,
            bert_random_init=True,
            global_cont_dim=16,
        )
        model.eval()
        with torch.no_grad():
            out = model(
                torch.randn(B, N, 4), torch.ones(B, N), global_cont=torch.randn(B, 16)
            )
        assert out.shape == (B, 1)

    def test_forward_with_global_token_requires_cond(self):
        from src.scripts.train import BertBaseline

        model = BertBaseline(
            input_dim=4,
            output_dim=1,
            pretrained_model_name_or_path="prajjwal1/bert-tiny",
            use_cls_token=False,
            bert_random_init=True,
            global_cont_dim=16,
        )
        model.eval()
        with pytest.raises(ValueError, match="global_cont is missing"):
            with torch.no_grad():
                _ = model(torch.randn(B, N, 4), torch.ones(B, N))
