#!/usr/bin/env python3
from __future__ import annotations

import math
import torch
from torch import nn

GLOBAL_LR_FEATURES_CHOICES = frozenset({"both", "instability", "mean_coverage"})


def _normalize_global_lr_features(features: str) -> str:
    f = str(features).strip().lower()
    if f not in GLOBAL_LR_FEATURES_CHOICES:
        raise ValueError(
            f"global_lr_features must be one of {sorted(GLOBAL_LR_FEATURES_CHOICES)}, got {features!r}"
        )
    return f


GLOBAL_LR_HEAD_TYPE_CHOICES = frozenset({"linear", "mlp"})


def _normalize_global_lr_head_type(head_type: str) -> str:
    t = str(head_type).strip().lower()
    if t not in GLOBAL_LR_HEAD_TYPE_CHOICES:
        raise ValueError(
            f"global_lr_head_type must be one of {sorted(GLOBAL_LR_HEAD_TYPE_CHOICES)}, got {head_type!r}"
        )
    return t


def _attach_global_lr_head(
    module: nn.Module,
    use_global_lr_head: bool,
    global_lr_features: str = "both",
    global_lr_raw_logit: bool = False,
    global_lr_normalize_logit: bool = False,
    global_lr_head_type: str = "linear",
    global_lr_mlp_hidden: int = 32,
) -> None:
    module.use_global_lr_head = bool(use_global_lr_head)
    module.global_lr_features = (
        _normalize_global_lr_features(global_lr_features) if use_global_lr_head else "both"
    )
    module.global_lr_raw_logit = bool(global_lr_raw_logit)
    module.global_lr_normalize_logit = bool(global_lr_normalize_logit)
    module.global_lr_head_type = (
        _normalize_global_lr_head_type(global_lr_head_type) if use_global_lr_head else "linear"
    )
    if use_global_lr_head:
        module.global_lr_head = GlobalLogisticRegressionHead(
            module.global_lr_features,
            global_lr_raw_logit=bool(global_lr_raw_logit),
            global_lr_normalize_logit=bool(global_lr_normalize_logit),
            head_type=module.global_lr_head_type,
            mlp_hidden=int(global_lr_mlp_hidden),
        )


class GlobalLogisticRegressionHead(nn.Module):
    """
    可训练融合头：在 Global head 的 logit 之后融合辅助特征，输出最终全局 logit（训练/推理时 prob=sigmoid(logit)）。

    head_type:
      linear — 单层逻辑回归 nn.Linear(n_in, 1)
      mlp    — MLP(n_in -> h -> h -> 1, GELU)，输出仍为 logit（与 focal loss 一致）

    features（第 1 维由 global_lr_normalize_logit / global_lr_raw_logit 控制）:
      global_lr_normalize_logit=True:
        第 1 维为 minmax01(global_logit)：用 training 集全局 min/max 线性缩放到 [0, 1]（见 set_global_logit_bounds）
      global_lr_raw_logit=True 且未 normalize:
        第 1 维为 raw global_logit
      否则（旧版）:
        第 1 维为 sigmoid(global_logit)
    """

    def __init__(
        self,
        features: str = "both",
        *,
        global_lr_raw_logit: bool = False,
        global_lr_normalize_logit: bool = False,
        head_type: str = "linear",
        mlp_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.features = _normalize_global_lr_features(features)
        self.global_lr_raw_logit = bool(global_lr_raw_logit)
        self.global_lr_normalize_logit = bool(global_lr_normalize_logit)
        self.head_type = _normalize_global_lr_head_type(head_type)
        n_in = {"both": 3, "instability": 2, "mean_coverage": 2}[self.features]
        if self.head_type == "linear":
            self.lr = nn.Linear(n_in, 1, bias=True)
            nn.init.zeros_(self.lr.weight)
            nn.init.zeros_(self.lr.bias)
            self.mlp: nn.Sequential | None = None
        else:
            self.lr = None
            h = max(4, int(mlp_hidden))
            self.mlp = nn.Sequential(
                nn.Linear(n_in, h, bias=True),
                nn.GELU(),
                nn.Linear(h, h, bias=True),
                nn.GELU(),
                nn.Linear(h, 1, bias=True),
            )
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    nn.init.zeros_(m.bias)
            last = self.mlp[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        if self.global_lr_normalize_logit:
            self.register_buffer("g_logit_min", torch.tensor(0.0, dtype=torch.float32))
            self.register_buffer("g_logit_max", torch.tensor(1.0, dtype=torch.float32))
            self.register_buffer("g_bounds_ready", torch.tensor(0.0, dtype=torch.float32))

    def set_global_logit_bounds(self, g_min: float, g_max: float) -> None:
        """由 training 全量 pre-GLR global_logit 的 min/max 设置 [0,1] 归一化上下界。"""
        if not self.global_lr_normalize_logit:
            return
        lo = float(g_min)
        hi = float(g_max)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"global_logit bounds must be finite, got min={g_min!r} max={g_max!r}")
        if hi <= lo:
            hi = lo + 1.0
        self.g_logit_min.fill_(lo)
        self.g_logit_max.fill_(hi)
        self.g_bounds_ready.fill_(1.0)

    def _minmax01_global(self, global_logit: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
        """用 training 全局 min/max 归一化到 [0, 1]；超出范围的值 clip。"""
        g = global_logit.reshape(-1).float()
        if g.numel() == 0:
            return g
        if float(self.g_bounds_ready.item()) < 0.5:
            raise RuntimeError(
                "global_lr_normalize_logit=True but global logit bounds not set; "
                "call set_global_logit_bounds() after scanning training split"
            )
        lo = self.g_logit_min
        hi = self.g_logit_max
        denom = hi - lo
        if float(denom.item()) <= float(eps):
            return torch.full_like(g, 0.5)
        out = (g - lo) / denom
        return out.clamp(0.0, 1.0)

    def _global_logit_feature(self, global_logit: torch.Tensor) -> torch.Tensor:
        if self.global_lr_normalize_logit:
            return self._minmax01_global(global_logit)
        g_feat = global_logit.reshape(-1)
        if not self.global_lr_raw_logit:
            g_feat = torch.sigmoid(g_feat)
        return g_feat

    def forward(
        self,
        global_logit: torch.Tensor,
        instability: torch.Tensor,
        mean_cov_log: torch.Tensor,
    ) -> torch.Tensor:
        g_feat = self._global_logit_feature(global_logit)
        parts: list[torch.Tensor] = [g_feat]
        if self.features in ("both", "instability"):
            parts.append(instability.reshape(-1))
        if self.features in ("both", "mean_coverage"):
            parts.append(mean_cov_log.reshape(-1))
        feats = torch.stack(parts, dim=1)
        if self.head_type == "linear":
            assert self.lr is not None
            return self.lr(feats).squeeze(1)
        assert self.mlp is not None
        return self.mlp(feats).reshape(-1)


def _transformer_pre_glr_global_logit(
    module: nn.Module,
    coverage_vec: torch.Tensor,
    mfe_scalar: torch.Tensor,
    mean_depth_scalar: torch.Tensor | None = None,
) -> torch.Tensor:
    """Transformer 全局头 out_global 输出（未经 Global LR head）。"""
    B = coverage_vec.size(0)
    x = coverage_vec.unsqueeze(2)
    x = module.in_proj(x)
    cls = module.cls.expand(B, -1, -1)
    cls = _inject_mfe_into_cls(module, cls, mfe_scalar)
    if getattr(module, "use_mean_depth", False):
        if mean_depth_scalar is None:
            raise ValueError("mean_depth_scalar required when use_mean_depth=True")
        if mean_depth_scalar.dim() == 1:
            mean_depth_scalar = mean_depth_scalar.unsqueeze(1)
        cls = cls + module.mean_depth_proj(mean_depth_scalar).unsqueeze(1)
    x = torch.cat([cls, x], dim=1)
    h = module.encoder(x)
    h_cls = h[:, 0, :]
    return module.out_global(h_cls).squeeze(1)


def _inject_mfe_into_cls(
    module: nn.Module,
    cls: torch.Tensor,
    mfe_scalar: torch.Tensor,
) -> torch.Tensor:
    """若 module.use_mfe 为 False，不向 CLS 注入 MFE（--no-mfe）。"""
    if not getattr(module, "use_mfe", True):
        return cls
    if mfe_scalar.dim() == 1:
        mfe_scalar = mfe_scalar.unsqueeze(1)
    return cls + module.mfe_proj(mfe_scalar).unsqueeze(1)


def _apply_global_lr_head(
    module: nn.Module,
    global_logit: torch.Tensor,
    instability_scalar: torch.Tensor | None,
    mean_cov_log_scalar: torch.Tensor | None,
) -> torch.Tensor:
    if not getattr(module, "use_global_lr_head", False):
        return global_logit
    if instability_scalar is None or mean_cov_log_scalar is None:
        raise ValueError("instability_scalar and mean_cov_log_scalar required when use_global_lr_head=True")
    return module.global_lr_head(global_logit, instability_scalar, mean_cov_log_scalar)


class DepthMfeLogit(nn.Module):
    """
    Implements:
      coverage_vec: (B, 180)
      mfe_scalar:   (B,) or (B,1)

      mfe_vec = W_mfe * mfe + b_mfe          # (B, 180)
      z = sum(w1 * mfe_vec) + sum(w2 * coverage_vec) + b   # (B,)

    This matches the spirit of z = w1·mfe + w2·coverage + b
    while keeping mfe mapped to 180 dims.
    """

    def __init__(
        self,
        dim: int = 180,
        *,
        use_mean_depth: bool = False,
        use_global_lr_head: bool = False,
        global_lr_features: str = "both",
        global_lr_raw_logit: bool = False,
        global_lr_normalize_logit: bool = False,
        global_lr_head_type: str = "linear",
        global_lr_mlp_hidden: int = 32,
        use_mfe: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_mean_depth = bool(use_mean_depth)
        self.use_mfe = bool(use_mfe)
        _attach_global_lr_head(
            self,
            use_global_lr_head,
            global_lr_features,
            global_lr_raw_logit,
            global_lr_normalize_logit,
            global_lr_head_type,
            global_lr_mlp_hidden,
        )

        if self.use_mfe:
            self.mfe_proj = nn.Linear(1, dim, bias=True)
            self.w1 = nn.Parameter(torch.zeros(dim))
            nn.init.normal_(self.mfe_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mfe_proj.bias)
            nn.init.normal_(self.w1, mean=0.0, std=0.02)
        self.w2 = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))
        if self.use_mean_depth:
            self.w_mean_depth = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        nn.init.normal_(self.w2, mean=0.0, std=0.02)

    def forward(
        self,
        coverage_vec: torch.Tensor,
        mfe_scalar: torch.Tensor,
        mean_depth_scalar: torch.Tensor | None = None,
        instability_scalar: torch.Tensor | None = None,
        mean_cov_log_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns logits (before sigmoid), shape (B,)
        """
        z = (coverage_vec * self.w2).sum(dim=1) + self.b
        if self.use_mfe:
            if mfe_scalar.dim() == 1:
                mfe_scalar = mfe_scalar.unsqueeze(1)
            mfe_vec = self.mfe_proj(mfe_scalar)
            z = z + (mfe_vec * self.w1).sum(dim=1)
        if self.use_mean_depth:
            if mean_depth_scalar is None:
                raise ValueError("mean_depth_scalar required when use_mean_depth=True")
            z = z + mean_depth_scalar.reshape(-1) * self.w_mean_depth
        return _apply_global_lr_head(self, z, instability_scalar, mean_cov_log_scalar)


class DepthMfeTransformerCLS(nn.Module):
    """
    Transformer encoder over coverage positions + CLS token.

    Inputs:
      coverage_vec: (B, dim)
      mfe_scalar:   (B,) or (B,1)

    We embed each position with a Linear(1 -> d_model), prepend a learnable CLS token,
    and add a projected MFE embedding to CLS (so MFE influences the global decision).

    Output:
      logits: (B,)
    """

    def __init__(
        self,
        dim: int = 180,
        *,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_mean_depth: bool = False,
        use_global_lr_head: bool = False,
        global_lr_features: str = "both",
        global_lr_raw_logit: bool = False,
        global_lr_normalize_logit: bool = False,
        global_lr_head_type: str = "linear",
        global_lr_mlp_hidden: int = 32,
        use_mfe: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.d_model = d_model
        self.use_mean_depth = bool(use_mean_depth)
        self.use_mfe = bool(use_mfe)
        _attach_global_lr_head(
            self,
            use_global_lr_head,
            global_lr_features,
            global_lr_raw_logit,
            global_lr_normalize_logit,
            global_lr_head_type,
            global_lr_mlp_hidden,
        )

        self.in_proj = nn.Linear(1, d_model, bias=True)
        if self.use_mfe:
            self.mfe_proj = nn.Linear(1, d_model, bias=True)
            nn.init.normal_(self.mfe_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mfe_proj.bias)
        if self.use_mean_depth:
            self.mean_depth_proj = nn.Linear(1, d_model, bias=True)
            nn.init.normal_(self.mean_depth_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mean_depth_proj.bias)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(
        self,
        coverage_vec: torch.Tensor,
        mfe_scalar: torch.Tensor,
        mean_depth_scalar: torch.Tensor | None = None,
        instability_scalar: torch.Tensor | None = None,
        mean_cov_log_scalar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B = coverage_vec.size(0)
        x = coverage_vec.unsqueeze(2)  # (B,dim,1)
        x = self.in_proj(x)  # (B,dim,d_model)

        cls = self.cls.expand(B, -1, -1)  # (B,1,d_model)
        cls = _inject_mfe_into_cls(self, cls, mfe_scalar)
        if self.use_mean_depth:
            if mean_depth_scalar is None:
                raise ValueError("mean_depth_scalar required when use_mean_depth=True")
            if mean_depth_scalar.dim() == 1:
                mean_depth_scalar = mean_depth_scalar.unsqueeze(1)
            cls = cls + self.mean_depth_proj(mean_depth_scalar).unsqueeze(1)

        x = torch.cat([cls, x], dim=1)  # (B,1+dim,d_model)
        h = self.encoder(x)  # (B,1+dim,d_model)
        h_cls = h[:, 0, :]  # (B,d_model)
        global_logits = self.out(h_cls).squeeze(1)  # (B,)
        return _apply_global_lr_head(self, global_logits, instability_scalar, mean_cov_log_scalar)


class DepthMfeTransformerCLSPos(nn.Module):
    """
    Same as DepthMfeTransformerCLS but also outputs per-position logits.

    Outputs:
      global_logits: (B,)
      pos_logits:    (B, dim)
    """

    def __init__(
        self,
        dim: int = 180,
        *,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pos_loss_weight_init: float = 1.0,
        use_mean_depth: bool = False,
        use_global_lr_head: bool = False,
        global_lr_features: str = "both",
        global_lr_raw_logit: bool = False,
        global_lr_normalize_logit: bool = False,
        global_lr_head_type: str = "linear",
        global_lr_mlp_hidden: int = 32,
        use_mfe: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.d_model = d_model
        self.use_mean_depth = bool(use_mean_depth)
        self.use_mfe = bool(use_mfe)
        _attach_global_lr_head(
            self,
            use_global_lr_head,
            global_lr_features,
            global_lr_raw_logit,
            global_lr_normalize_logit,
            global_lr_head_type,
            global_lr_mlp_hidden,
        )

        self.in_proj = nn.Linear(1, d_model, bias=True)
        if self.use_mfe:
            self.mfe_proj = nn.Linear(1, d_model, bias=True)
            nn.init.normal_(self.mfe_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mfe_proj.bias)
        if self.use_mean_depth:
            self.mean_depth_proj = nn.Linear(1, d_model, bias=True)
            nn.init.normal_(self.mean_depth_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mean_depth_proj.bias)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_global = nn.Linear(d_model, 1)
        self.out_pos = nn.Linear(d_model, 1)

        # Learnable weight for position loss (kept positive via exp()).
        # This lets training automatically balance global vs position supervision.
        init = float(pos_loss_weight_init)
        init = 1e-6 if init <= 0 else init
        self.log_pos_loss_weight = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float32))

    def pos_loss_weight(self) -> torch.Tensor:
        # strictly positive scalar
        return torch.exp(self.log_pos_loss_weight)

    def forward(
        self,
        coverage_vec: torch.Tensor,
        mfe_scalar: torch.Tensor,
        mean_depth_scalar: torch.Tensor | None = None,
        instability_scalar: torch.Tensor | None = None,
        mean_cov_log_scalar: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = coverage_vec.size(0)
        x = coverage_vec.unsqueeze(2)  # (B,dim,1)
        x = self.in_proj(x)  # (B,dim,d_model)

        cls = self.cls.expand(B, -1, -1)  # (B,1,d_model)
        cls = _inject_mfe_into_cls(self, cls, mfe_scalar)
        if self.use_mean_depth:
            if mean_depth_scalar is None:
                raise ValueError("mean_depth_scalar required when use_mean_depth=True")
            if mean_depth_scalar.dim() == 1:
                mean_depth_scalar = mean_depth_scalar.unsqueeze(1)
            cls = cls + self.mean_depth_proj(mean_depth_scalar).unsqueeze(1)

        x = torch.cat([cls, x], dim=1)  # (B,1+dim,d_model)
        h = self.encoder(x)  # (B,1+dim,d_model)
        h_cls = h[:, 0, :]  # (B,d_model)
        h_pos = h[:, 1:, :]  # (B,dim,d_model)

        global_logits = self.out_global(h_cls).squeeze(1)  # (B,)
        global_logits = _apply_global_lr_head(self, global_logits, instability_scalar, mean_cov_log_scalar)
        pos_logits = self.out_pos(h_pos).squeeze(2)  # (B,dim)
        return global_logits, pos_logits


class DepthMfeTransformerCLSPosMature(nn.Module):
    """
    Transformer+CLS with two per-position heads:
      - pos_logits:    (B, dim)  (your existing precursor-length mask target)
      - mature_logits: (B, dim)  (mature region mask target)

    Also includes learnable positive weights for each auxiliary loss.
    """

    def __init__(
        self,
        dim: int = 180,
        *,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pos_loss_weight_init: float = 1.0,
        mature_loss_weight_init: float = 1.0,
        use_mean_depth: bool = False,
        use_global_lr_head: bool = False,
        global_lr_features: str = "both",
        global_lr_raw_logit: bool = False,
        global_lr_normalize_logit: bool = False,
        global_lr_head_type: str = "linear",
        global_lr_mlp_hidden: int = 32,
        use_mfe: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.d_model = d_model
        self.use_mean_depth = bool(use_mean_depth)
        self.use_mfe = bool(use_mfe)
        _attach_global_lr_head(
            self,
            use_global_lr_head,
            global_lr_features,
            global_lr_raw_logit,
            global_lr_normalize_logit,
            global_lr_head_type,
            global_lr_mlp_hidden,
        )

        self.in_proj = nn.Linear(1, d_model, bias=True)
        if self.use_mfe:
            self.mfe_proj = nn.Linear(1, d_model, bias=True)
            nn.init.normal_(self.mfe_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mfe_proj.bias)
        if self.use_mean_depth:
            self.mean_depth_proj = nn.Linear(1, d_model, bias=True)
            nn.init.normal_(self.mean_depth_proj.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.mean_depth_proj.bias)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_global = nn.Linear(d_model, 1)
        self.out_pos = nn.Linear(d_model, 1)
        self.out_mature = nn.Linear(d_model, 1)

        # Learnable weights (kept positive via exp()).
        def _init_logw(v: float) -> torch.Tensor:
            v = float(v)
            v = 1e-6 if v <= 0 else v
            return torch.tensor(math.log(v), dtype=torch.float32)

        self.log_pos_loss_weight = nn.Parameter(_init_logw(pos_loss_weight_init))
        self.log_mature_loss_weight = nn.Parameter(_init_logw(mature_loss_weight_init))

    def pos_loss_weight(self) -> torch.Tensor:
        return torch.exp(self.log_pos_loss_weight)

    def mature_loss_weight(self) -> torch.Tensor:
        return torch.exp(self.log_mature_loss_weight)

    def forward(
        self,
        coverage_vec: torch.Tensor,
        mfe_scalar: torch.Tensor,
        mean_depth_scalar: torch.Tensor | None = None,
        instability_scalar: torch.Tensor | None = None,
        mean_cov_log_scalar: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = coverage_vec.size(0)
        x = coverage_vec.unsqueeze(2)  # (B,dim,1)
        x = self.in_proj(x)  # (B,dim,d_model)

        cls = self.cls.expand(B, -1, -1)  # (B,1,d_model)
        cls = _inject_mfe_into_cls(self, cls, mfe_scalar)
        if self.use_mean_depth:
            if mean_depth_scalar is None:
                raise ValueError("mean_depth_scalar required when use_mean_depth=True")
            if mean_depth_scalar.dim() == 1:
                mean_depth_scalar = mean_depth_scalar.unsqueeze(1)
            cls = cls + self.mean_depth_proj(mean_depth_scalar).unsqueeze(1)

        x = torch.cat([cls, x], dim=1)  # (B,1+dim,d_model)
        h = self.encoder(x)  # (B,1+dim,d_model)
        h_cls = h[:, 0, :]  # (B,d_model)
        h_pos = h[:, 1:, :]  # (B,dim,d_model)

        global_logits = self.out_global(h_cls).squeeze(1)  # (B,)
        global_logits = _apply_global_lr_head(self, global_logits, instability_scalar, mean_cov_log_scalar)
        pos_logits = self.out_pos(h_pos).squeeze(2)  # (B,dim)
        mature_logits = self.out_mature(h_pos).squeeze(2)  # (B,dim)
        return global_logits, pos_logits, mature_logits


class DepthOnlyLogit(nn.Module):
    """
    Implements:
      coverage_vec: (B, dim)
      z = sum(w * coverage_vec) + b   # (B,)
    """

    def __init__(self, dim: int = 180):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.w, mean=0.0, std=0.02)

    def forward(self, coverage_vec: torch.Tensor) -> torch.Tensor:
        return (coverage_vec * self.w).sum(dim=1) + self.b

