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
        assert (out >= 0).all(), "output_positive=True should produce non-negative values"

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
            out = model(torch.randn(B, N, 4), torch.ones(B, N), global_cont=torch.randn(B, 16))
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
