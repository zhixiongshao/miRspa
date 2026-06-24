#!/usr/bin/env python3
"""茎区/序列覆盖度不稳定性 S（与 plot_top20_depth_variance_stability_hist 第二轨一致）。"""
from __future__ import annotations

import numpy as np


def top_k_depth_std(x: np.ndarray, *, k: int = 20) -> float:
    v = np.asarray(x, dtype=np.float64).ravel()
    n = int(v.size)
    if n == 0:
        return float("nan")
    kk = min(int(k), n)
    if kk == 1:
        return 0.0
    part = np.partition(v, -kk)[-kk:]
    return float(np.std(part, ddof=0))


def minmax_normalize_per_sequence(x: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(x, dtype=np.float64).ravel(), 0.0)
    mn = float(np.min(p))
    mx = float(np.max(p))
    if mx <= mn:
        return np.zeros_like(p)
    return (p - mn) / (mx - mn)


def top_k_minmax_norm_std(x: np.ndarray, *, k: int = 20) -> float:
    v = minmax_normalize_per_sequence(x)
    return top_k_depth_std(v, k=k)


def coverage_all_zero(x: np.ndarray) -> bool:
    v = np.asarray(x, dtype=np.float64).ravel()
    if v.size == 0:
        return True
    return bool(np.all(v <= 0.0))


def instability_s_minmax_norm_std(x: np.ndarray, *, top_k: int = 20) -> float:
    """条内 min–max 后 top-K 深度标准差；全零或空向量返回 nan。"""
    if coverage_all_zero(x):
        return float("nan")
    s = top_k_minmax_norm_std(x, k=top_k)
    return float(s) if np.isfinite(s) else float("nan")


DEFAULT_MEAN_COV_DIVISOR = 10000.0


def mean_coverage_log1p(mu_raw: float, *, divisor: float = DEFAULT_MEAN_COV_DIVISOR) -> float:
    """log1p(mean_raw / divisor)，供 Global LR / MLP 头的 mean_coverage 维使用。"""
    d = float(divisor)
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError(f"mean_cov divisor must be finite and > 0, got {divisor!r}")
    return float(np.log1p(max(float(mu_raw), 0.0) / d))


def mean_coverage_log1p_div10k(mu_raw: float) -> float:
    """向后兼容：等价于 mean_coverage_log1p(mu_raw, divisor=10000)。"""
    return mean_coverage_log1p(mu_raw, divisor=DEFAULT_MEAN_COV_DIVISOR)


def resolve_mean_cov_divisor(
    *,
    ckpt_args: dict | None = None,
    override: float | None = None,
) -> float:
    """
  推理时 mean_coverage 预处理除数：
  CLI --mean-cov-divisor > ckpt args['mean_cov_divisor'] > 默认 10000。
  不修改 model state_dict，仅影响进 MLP 的第 3 维标量特征。
    """
    if override is not None:
        d = float(override)
    elif ckpt_args and ckpt_args.get("mean_cov_divisor") is not None:
        d = float(ckpt_args["mean_cov_divisor"])
    else:
        d = DEFAULT_MEAN_COV_DIVISOR
    if not np.isfinite(d) or d <= 0.0:
        raise ValueError(f"mean_cov_divisor must be finite and > 0, got {d!r}")
    return d
