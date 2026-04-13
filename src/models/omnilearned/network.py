import os
import torch
import torch.nn as nn


from .layers import (
    NoScaleDropout,
    InteractionBlock,
    LocalEmbeddingBlock,
    MLP,
    AttBlock,
    DynamicTanh,
    InputBlock,
    TokenAttBlock,
)
from .diffusion import MPFourier, perturb, get_logsnr_alpha_sigma


class PET2(nn.Module):
    def __init__(
        self,
        input_dim,
        use_int=True,
        local_int=True,
        int_type="lhc",
        conditional=False,
        cond_dim=3,
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        mode="classifier",
        num_classes=2,
        num_gen_classes=1,
        base_dim=128,
        num_transformers=2,
        num_transformers_head=2,
        num_tokens=4,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.0,
        attn_drop=0.0,
        feature_drop=0.0,
        K=15,
        skip=False,
        num_coord=3,
    ):
        super().__init__()
        self.mode = mode
        if self.mode not in [
            "classifier",
            "generator",
            "regression",
            "segmentation",
            "ftag",
            "pretrain",
        ]:
            raise ValueError(f"Mode '{self.mode}' not supported.")

        self.body = PET_body(
            input_dim,
            base_dim,
            num_transformers=num_transformers,
            num_transf_local=num_transformers_head,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            act_layer=act_layer,
            mlp_drop=mlp_drop,
            attn_drop=attn_drop,
            feature_drop=feature_drop,
            num_tokens=num_tokens,
            K=K,
            use_int=use_int,
            local_int=local_int,
            int_type=int_type,
            conditional=conditional,
            cond_dim=cond_dim,
            pid=pid,
            pid_dim=pid_dim,
            add_info=add_info,
            add_dim=add_dim,
            use_time=self.mode in ["generator", "pretrain"],
            skip=skip,
            num_coord=num_coord,
        )

        self.num_add = self.body.num_add
        self.num_tokens = num_tokens
        self.classifier = None
        self.generator = None

        use_classifier = self.mode in ["classifier", "ftag", "regression", "pretrain"]
        use_generator = self.mode in ["generator", "segmentation", "ftag", "pretrain"]
        if use_classifier:
            self.classifier = PET_classifier(
                base_dim,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                num_tokens=num_tokens,
                num_classes=num_classes,
            )

        if use_generator:
            self.generator = PET_generator(
                (
                    input_dim if num_gen_classes == 1 else num_gen_classes
                ),  # diffusion or segmentation
                base_dim,
                num_transformers=num_transformers_head,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_drop=mlp_drop,
                attn_drop=attn_drop,
                num_tokens=num_tokens,
                num_add=self.num_add,
                num_classes=num_classes,
                skip_pid=(num_classes == 1) | (self.mode == "segmentation"),
            )

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def no_weight_decay(self):
        # Specify parameters that should not be decayed
        return {"norm", "token"}

    def forward(self, x, y, cond=None, pid=None, add_info=None):
        y_pred, y_perturb, z_pred, v, v_weight, x_body, z_body = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

        time = torch.rand(size=(x.shape[0],)).to(x.device)
        _, alpha, sigma = get_logsnr_alpha_sigma(time)

        if self.mode in ["generator", "pretrain"]:
            z, v, v_weight = perturb(x, time)
            z_body = self.body(z, cond, pid, add_info, time)
            z_pred = self.generator(z_body, y)

        if self.mode in [
            "classifier",
            "regression",
            "segmentation",
            "pretrain",
            "ftag",
        ]:
            x_body = self.body(x, cond, pid, add_info, torch.zeros_like(time))

            if self.mode in ["classifier", "pretrain", "regression", "ftag"]:
                y_pred = self.classifier(x_body)
            if self.mode == "pretrain":
                y_perturb = self.classifier(z_body)
            if self.mode == "ftag" or self.mode == "segmentation":
                z_pred = self.generator(x_body, y)

        return {
            "y_pred": y_pred,
            "y_perturb": y_perturb,
            "z_pred": z_pred,
            "v": v,
            "v_weight": v_weight,
            "x_body": x_body,
            "z_body": z_body,
            "alpha": alpha**2,
        }


class PET_classifier(nn.Module):
    def __init__(
        self,
        base_dim,
        num_transformers=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        num_tokens=4,
        num_classes=2,
    ):
        super().__init__()
        self.num_tokens = num_tokens

        self.in_blocks = nn.ModuleList(
            [
                TokenAttBlock(
                    dim=base_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    num_tokens=self.num_tokens,
                    skip=False,
                )
                for _ in range(num_transformers)
            ]
        )

        self.fc = MLP(
            base_dim * self.num_tokens,
            int(mlp_ratio * self.num_tokens * base_dim),
            act_layer=act_layer,
            drop=mlp_drop,
            norm_layer=norm_layer,
        )

        self.out = nn.Linear(self.num_tokens * base_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x):
        B = x.shape[0]
        mask = x[:, self.num_tokens :, 2:3] != 0
        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask)

        x = self.fc(x[:, : self.num_tokens].reshape(B, -1))
        debug_linear = os.environ.get("OMNILEARNED_DEBUG_LINEAR", "0") == "1"
        if debug_linear:
            print(
                "[PET_classifier.out] "
                f"x.shape={tuple(x.shape)}, x.dtype={x.dtype}, x.device={x.device}, "
                f"w.shape={tuple(self.out.weight.shape)}, w.dtype={self.out.weight.dtype}, "
                f"w.device={self.out.weight.device}, b.shape={None if self.out.bias is None else tuple(self.out.bias.shape)}"
            )
            print(
                "[PET_classifier.out] "
                f"x_has_nan={torch.isnan(x).any().item()}, x_has_inf={torch.isinf(x).any().item()}, "
                f"w_has_nan={torch.isnan(self.out.weight).any().item()}, "
                f"w_has_inf={torch.isinf(self.out.weight).any().item()}"
            )

        try:
            return self.out(x)
        except Exception as e:
            print(
                "[PET_classifier.out][EXCEPTION] "
                f"x.shape={tuple(x.shape)}, x.dtype={x.dtype}, x.device={x.device}, "
                f"w.shape={tuple(self.out.weight.shape)}, w.dtype={self.out.weight.dtype}, "
                f"w.device={self.out.weight.device}, b.shape={None if self.out.bias is None else tuple(self.out.bias.shape)}"
            )
            print(
                "[PET_classifier.out][EXCEPTION] "
                f"x_min={x.detach().min().item() if x.numel() > 0 else 'NA'}, "
                f"x_max={x.detach().max().item() if x.numel() > 0 else 'NA'}, "
                f"x_has_nan={torch.isnan(x).any().item()}, x_has_inf={torch.isinf(x).any().item()}"
            )
            raise e


class PET_generator(nn.Module):
    def __init__(
        self,
        output_size,
        base_dim,
        num_transformers=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        num_tokens=4,
        num_add=1,
        num_classes=2,
        skip_pid=False,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_add = num_add
        self.num_classes = num_classes
        self.skip_pid = skip_pid

        if not self.skip_pid:
            self.pid_embed = nn.Sequential(
                nn.Embedding(num_classes, base_dim),
                MLP(
                    base_dim,
                    int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    act_layer=act_layer,
                    drop=mlp_drop,
                ),
            )
            self.num_add = self.num_add + 1

        self.in_blocks = nn.ModuleList(
            [
                AttBlock(
                    dim=base_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    num_tokens=num_tokens,
                    skip=False,
                    use_int=False,
                )
                for _ in range(num_transformers)
            ]
        )

        self.fc = nn.Sequential(
            MLP(
                base_dim,
                int(mlp_ratio * base_dim),
                act_layer=act_layer,
                drop=mlp_drop,
            ),
        )

        self.out = nn.Linear(base_dim, output_size)

        self.initialize_weights()

    def initialize_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x, y=None):
        # Add tokens and label embedding
        mask = x[:, :, 2:3] != 0

        if not self.skip_pid and y is not None:
            mask = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
            x = torch.cat([self.pid_embed(y).unsqueeze(1), x], 1) * mask

        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask)
        x = (
            self.fc(x[:, self.num_add + self.num_tokens :])
            * mask[:, self.num_add + self.num_tokens :]
        )
        return self.out(x) * mask[:, self.num_add + self.num_tokens :]


class PET_body(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
        num_transformers=2,
        num_transf_local=2,
        num_heads=4,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        mlp_drop=0.1,
        attn_drop=0.1,
        feature_drop=0.0,
        num_tokens=4,
        K=10,
        use_int=True,
        local_int=True,
        int_type="lhc",
        conditional=False,
        cond_dim=3,
        pid=False,
        pid_dim=9,
        add_info=False,
        add_dim=4,
        use_time=False,
        skip=False,
        num_coord=3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.use_int = use_int
        self.use_time = use_time
        self.conditional = conditional
        self.pid = pid
        self.add_info = add_info
        self.skip = skip
        self.num_coord = num_coord
        self.embed = InputBlock(
            in_features=input_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        if self.use_int:
            self.interaction = InteractionBlock(
                hidden_features=base_dim,
                out_features=num_heads,
                # mlp_drop=mlp_drop,
                act_layer=act_layer,
                norm_layer=norm_layer,
                int_type=int_type,
            )

        self.local_physics = LocalEmbeddingBlock(
            in_features=input_dim,
            hidden_features=mlp_ratio * base_dim,
            out_features=base_dim,
            act_layer=act_layer,
            mlp_drop=mlp_drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            K=K,
            num_heads=num_heads,
            local_int=local_int,
            int_type=int_type,
            num_transformers=num_transf_local,
        )

        self.num_add = 0
        if self.conditional:
            self.cond_embed = nn.Sequential(
                MLP(
                    in_features=cond_dim,
                    hidden_features=base_dim,
                    out_features=base_dim,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
            )
            self.num_add += 1

        if self.use_time:
            # Time embedding module for diffusion timesteps
            self.MPFourier = MPFourier(base_dim)
            self.time_embed = MLP(
                in_features=base_dim,
                hidden_features=int(mlp_ratio * base_dim),
                out_features=base_dim,
                norm_layer=norm_layer,
                act_layer=act_layer,
                bias=False,
            )
            self.num_add += 1

        if self.add_info:
            self.add_embed = nn.Sequential(
                MLP(
                    in_features=add_dim,
                    hidden_features=int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    bias=False,
                ),
                NoScaleDropout(feature_drop),
            )

        if self.pid:
            # Will assume PIDs are just a list of integers and use the embedding layer, notice that zero_pid_idx is used to map zero-padded entries
            self.pid_embed = nn.Sequential(
                nn.Embedding(pid_dim, base_dim, padding_idx=0),
                NoScaleDropout(feature_drop),
            )

        self.num_tokens = num_tokens
        self.token = nn.Parameter(1e-3 * torch.ones(1, self.num_tokens, base_dim))

        self.num_heads = num_heads

        if self.skip:
            self.in_blocks = nn.ModuleList(
                [
                    AttBlock(
                        dim=base_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        mlp_drop=mlp_drop,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        num_tokens=num_tokens + self.num_add,
                        skip=False,
                        use_int=use_int,
                    )
                    for _ in range(num_transformers // 2)
                ]
            )

            self.out_blocks = nn.ModuleList(
                [
                    AttBlock(
                        dim=base_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        mlp_drop=mlp_drop,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        num_tokens=num_tokens + self.num_add,
                        skip=True,
                        use_int=use_int,
                    )
                    for _ in range(num_transformers // 2)
                ]
            )

        else:
            self.in_blocks = nn.ModuleList(
                [
                    AttBlock(
                        dim=base_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop,
                        mlp_drop=mlp_drop,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        num_tokens=num_tokens + self.num_add,
                        skip=False,
                        use_int=use_int,
                    )
                    for _ in range(num_transformers)
                ]
            )

        self.norm = norm_layer(base_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.token, mean=0.0, std=0.02, a=-2.0, b=2.0)

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

    def forward(self, x, cond=None, pid=None, add_info=None, time=None):
        B = x.shape[0]
        mask = x[:, :, 2:3] != 0
        token = self.token.expand(B, -1, -1)

        x_embed, x = self.embed(x, mask)

        # Move away zero-padded entries
        coord_shift = 999.0 * (~mask).float()
        points = coord_shift + x[:, :, : self.num_coord]
        local_features, indices = self.local_physics(points, x, mask)

        x_int = None
        if self.use_int:
            x_int = self.interaction(x, mask)

        # Combine local + global info
        x = x_embed + local_features
        # Add classification tokens

        if pid is not None and self.pid:
            # Encode the PID info
            x = x + self.pid_embed(pid) * mask
        if add_info is not None and self.add_info:
            x = x + self.add_embed(add_info) * mask

        if cond is not None and self.conditional:
            # Conditional information: jet level quantities for example
            x = torch.cat([self.cond_embed(cond).unsqueeze(1), x], 1)

        if self.use_time and time is not None:
            # Create time token
            x = torch.cat([self.time_embed(self.MPFourier(time)).unsqueeze(1), x], 1)

        x = torch.cat([token, x], 1)

        # Create a new mask based on the updated point cloud with additional tokens
        mask = x[:, :, 2:3] != 0

        attn_mask = mask.float() @ mask.float().transpose(-1, -2)
        attn_mask = ~(attn_mask.bool()).repeat_interleave(self.num_heads, dim=0)
        attn_mask = attn_mask.float() * -1e9

        if self.use_int:
            # Add the information of the interaction matrix
            attn_mask[
                :, self.num_tokens + self.num_add :, self.num_tokens + self.num_add :
            ] = (
                x_int
                + attn_mask[
                    :,
                    self.num_tokens + self.num_add :,
                    self.num_tokens + self.num_add :,
                ]
            )

        skips = []
        for ib, blk in enumerate(self.in_blocks):
            x = blk(x, mask=mask, attn_mask=attn_mask)
            skips.append(x)
        if self.skip:
            for ib, blk in enumerate(self.out_blocks):
                x = blk(x, mask=mask, attn_mask=attn_mask, skip=skips.pop())

        x = self.norm(x) * mask
        return x


class MLPGEN(nn.Module):
    def __init__(
        self,
        input_dim,
        base_dim,
        mlp_ratio=2,
        norm_layer=DynamicTanh,
        act_layer=nn.GELU,
        num_layers=3,
        mlp_drop=0.0,
        conditional=False,
        cond_dim=1,
    ):
        super().__init__()
        self.embed = MLP(
            in_features=input_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim // 2,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        self.cond = MLP(
            in_features=cond_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim // 2,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )

        self.in_blocks = nn.ModuleList(
            [
                MLP(
                    in_features=base_dim,
                    hidden_features=int(mlp_ratio * base_dim),
                    out_features=base_dim,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for _ in range(num_layers)
            ]
        )

        # Time embedding module for diffusion timesteps
        self.MPFourier = MPFourier(base_dim)
        self.time_embed = MLP(
            in_features=base_dim,
            hidden_features=int(mlp_ratio * base_dim),
            out_features=base_dim // 2,
            act_layer=act_layer,
        )

        self.out = nn.Linear(base_dim, input_dim)

    def forward(self, z, time, cond=None):
        z = self.embed(z)
        z = z + self.cond(cond)
        z = torch.cat([self.time_embed(self.MPFourier(time)), z], 1)
        for ib, blk in enumerate(self.in_blocks):
            z = blk(z)
        z = self.out(z)
        return z
