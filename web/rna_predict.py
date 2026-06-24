#!/usr/bin/env python3
"""miRspa-RNA 单条推理（与 eval_from_full_tsv.py 特征/前向逻辑一致）。"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

WEB_DIR = Path(__file__).resolve().parent


def _repo_root() -> Path:
    start = WEB_DIR
    for d in [start.parent, start, *start.parents]:
        if (d / "model_depth_mfe.py").is_file():
            return d
    raise RuntimeError(
        "找不到 model_depth_mfe.py；请确认仓库根目录含该文件（与 web/ 同级）。"
    )


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coverage_instability import (  # noqa: E402
    instability_s_minmax_norm_std,
    mean_coverage_log1p,
    resolve_mean_cov_divisor,
)
from model_depth_mfe import DepthMfeTransformerCLSPosMature  # noqa: E402
from mature_region_detect import find_mature_regions, mature_regions_to_genomic  # noqa: E402

DEFAULT_RNA_CHECKPOINT = WEB_DIR / "models/miRspa-RNA/best.pt"

RNA_STATE_LOCK = threading.Lock()
RNA_STATE: dict[str, Any] = {
    "ready": False,
    "model": None,
    "device": None,
    "ckpt_args": None,
    "checkpoint": None,
    "dim": 180,
}


def _pad_trunc_back(x: np.ndarray, out_len: int, *, pad_value: float = -1.0) -> np.ndarray:
    if x.size == out_len:
        return x
    if x.size > out_len:
        return x[:out_len]
    pad = np.full((out_len - x.size,), float(pad_value), dtype=np.float32)
    return np.concatenate([x, pad], axis=0)


def _preprocess_reads(
    reads: list[float],
    out_len: int,
    *,
    log1p: bool,
    standardize: bool,
) -> np.ndarray:
    x = np.asarray(reads, dtype=np.float32)
    if log1p:
        x = np.log1p(x)
    if standardize:
        mu = float(x.mean()) if x.size else 0.0
        sd = float(x.std(ddof=0)) if x.size else 0.0
        if sd > 0:
            x = (x - mu) / sd
        else:
            x = x - mu
        if x.size:
            mn = float(x.min())
            mx = float(x.max())
            if mx > mn:
                x = (x - mn) / (mx - mn)
            else:
                x = x * 0.0
    return _pad_trunc_back(x, out_len, pad_value=-1.0)


def _reads_log1p_from_ckpt(ckpt_args: dict) -> bool:
    if not ckpt_args:
        return False
    if "log1p" in ckpt_args:
        return bool(ckpt_args["log1p"])
    return not bool(ckpt_args.get("no_log1p", False))


def _normalize_mfe(mfe_raw: float, mfe_min: float, mfe_max: float) -> float:
    m_abs = abs(float(mfe_raw))
    if mfe_max > mfe_min:
        return float((m_abs - mfe_min) / (mfe_max - mfe_min))
    return float(m_abs)


def mature_region_from_prediction(
    *,
    mature_prob: list[float],
    pos_prob: list[float],
    reads: list[float],
    stem_len: int,
    stem_start_1b: int,
    stem_end_1b: int,
    strand: str,
    chrom: str,
    stem_rna: str,
) -> dict[str, Any]:
    """调用与 Pancancer export 脚本一致的 find_mature_regions，返回 1–2 段 5p/3p。"""
    raw_regions = find_mature_regions(
        np.asarray(mature_prob, dtype=np.float64),
        np.asarray(pos_prob, dtype=np.float64),
        int(stem_len),
        reads=reads,
    )
    mature_regions = mature_regions_to_genomic(
        raw_regions,
        stem_start_1b=int(stem_start_1b),
        stem_end_1b=int(stem_end_1b),
        strand=str(strand),
        chrom=str(chrom),
        stem_rna=stem_rna,
    )

    primary = mature_regions[0] if mature_regions else None
    return {
        "mature_regions": mature_regions,
        "mature_stem_start_idx": primary["stem_start_idx"] if primary else None,
        "mature_stem_end_idx": primary["stem_end_idx"] if primary else None,
        "mature_start_1b": primary["genomic_start_1b"] if primary else None,
        "mature_end_1b": primary["genomic_end_1b"] if primary else None,
        "mature_sequence_rna": primary["sequence_rna"] if primary else None,
        "mature_prob_mean": primary["mean_prob"] if primary else None,
    }


def parse_depths_payload(raw: object, *, expected_len: int | None = None) -> list[float]:
    """解析用户覆盖度：JSON 数组，或逗号/空白分隔字符串。"""
    if raw is None:
        raise ValueError("缺少覆盖度 depths / reads")
    if isinstance(raw, list):
        vals = [float(v) for v in raw]
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise ValueError("覆盖度为空")
        if s.startswith("["):
            import json

            arr = json.loads(s)
            if not isinstance(arr, list):
                raise ValueError("depths JSON 必须是数组")
            vals = [float(v) for v in arr]
        else:
            parts = [p for p in s.replace(",", " ").split() if p]
            vals = [float(p) for p in parts]
    else:
        raise ValueError("depths 类型无效（需要数组或字符串）")
    if any(v < 0 for v in vals):
        raise ValueError("覆盖度不能为负数")
    if expected_len is not None and len(vals) != int(expected_len):
        raise ValueError(
            f"覆盖度位数 {len(vals)} 与区间长度 {expected_len} 不一致 "
            f"（1-based 闭区间应提供 end-start+1 个深度）"
        )
    return vals


def ensure_rna_model(checkpoint: Path, device_str: str) -> dict[str, Any]:
    with RNA_STATE_LOCK:
        ckpt_path = Path(checkpoint).resolve()
        if RNA_STATE["ready"] and Path(RNA_STATE["checkpoint"]).resolve() == ckpt_path:
            return RNA_STATE

        dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")

        ckpt_args = ckpt.get("args") or {}
        arch = str(ckpt_args.get("arch", "linear"))
        if arch != "transformer_cls_pos_mature":
            raise RuntimeError(f"不支持的 RNA arch: {arch!r}")

        dim = int(ckpt_args.get("dim", 180))
        use_mean_depth = bool(ckpt_args.get("use_mean_depth", False))
        use_global_lr_head = bool(ckpt_args.get("global_lr_head", False))
        global_lr_features = str(ckpt_args.get("global_lr_features", "both"))
        global_lr_raw_logit = bool(ckpt_args.get("global_lr_raw_logit", False))
        global_lr_normalize_logit = bool(ckpt_args.get("global_lr_normalize_logit", False))
        global_lr_head_type = "mlp" if bool(ckpt_args.get("global_lr_mlp", False)) else str(
            ckpt_args.get("global_lr_head_type", "linear")
        )
        global_lr_mlp_hidden = int(ckpt_args.get("global_lr_mlp_hidden", 32))
        use_mfe = not bool(ckpt_args.get("no_mfe", False))

        model = DepthMfeTransformerCLSPosMature(
            dim=dim,
            use_mean_depth=use_mean_depth,
            use_global_lr_head=use_global_lr_head,
            global_lr_features=global_lr_features,
            global_lr_raw_logit=global_lr_raw_logit,
            global_lr_normalize_logit=global_lr_normalize_logit,
            global_lr_head_type=global_lr_head_type,
            global_lr_mlp_hidden=global_lr_mlp_hidden,
            use_mfe=use_mfe,
        ).to(dev)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        RNA_STATE.update(
            {
                "ready": True,
                "model": model,
                "device": dev,
                "ckpt_args": ckpt_args,
                "checkpoint": str(ckpt_path),
                "dim": dim,
                "arch": arch,
                "use_mfe": use_mfe,
                "use_mean_depth": use_mean_depth,
                "use_global_lr_head": use_global_lr_head,
                "global_lr_features": global_lr_features,
                "instability_top_k": int(ckpt_args.get("instability_top_k", 20)),
            }
        )
        return RNA_STATE


def predict_rna(
    *,
    reads: list[float],
    mfe_raw: float,
    mean_cov_divisor: float,
    checkpoint: Path,
    device_str: str = "cpu",
) -> dict[str, Any]:
    state = ensure_rna_model(checkpoint, device_str)
    ckpt_args = state["ckpt_args"] or {}
    dim = int(state["dim"])
    reads_log1p = _reads_log1p_from_ckpt(ckpt_args)
    standardize = not bool(ckpt_args.get("no_standardize", False))
    mfe_min = float(ckpt_args.get("mfe_abs_min", 0.0))
    mfe_max = float(ckpt_args.get("mfe_abs_max", 144.7))
    top_k = int(state.get("instability_top_k", 20))

    divisor = resolve_mean_cov_divisor(ckpt_args=ckpt_args, override=float(mean_cov_divisor))

    mu_raw = float(np.mean(reads)) if reads else 0.0
    raw_inst = instability_s_minmax_norm_std(np.asarray(reads, dtype=np.float64), top_k=top_k)
    instability = float("nan") if not np.isfinite(raw_inst) else float(raw_inst)
    mean_cov_log = mean_coverage_log1p(mu_raw, divisor=divisor)

    x_cov = _preprocess_reads(reads, dim, log1p=reads_log1p, standardize=standardize)
    mfe_norm = _normalize_mfe(mfe_raw, mfe_min, mfe_max) if state["use_mfe"] else 0.0

    device = state["device"]
    model = state["model"]
    xb = torch.from_numpy(x_cov[None, ...].astype(np.float32, copy=False)).to(device)
    mfb = torch.from_numpy(np.asarray([mfe_norm], dtype=np.float32)).to(device)

    kw: dict[str, torch.Tensor] = {}
    if state["use_global_lr_head"]:
        inst_v = 0.0 if not np.isfinite(instability) else float(instability)
        kw["instability_scalar"] = torch.from_numpy(np.asarray([inst_v], dtype=np.float32)).to(device)
        kw["mean_cov_log_scalar"] = torch.from_numpy(
            np.asarray([mean_cov_log], dtype=np.float32)
        ).to(device)

    with torch.no_grad():
        if state["use_mean_depth"]:
            mddb = torch.from_numpy(np.asarray([mu_raw], dtype=np.float32)).to(device)
            glog, plog, mlog = model(xb, mfb, mddb, **kw)
        else:
            glog, plog, mlog = model(xb, mfb, **kw)
        prob = float(torch.sigmoid(glog).detach().cpu().numpy().reshape(-1)[0])
        pos_prob = torch.sigmoid(plog).detach().cpu().numpy().astype(np.float32).reshape(-1).tolist()
        mature_prob = (
            torch.sigmoid(mlog).detach().cpu().numpy().astype(np.float32).reshape(-1).tolist()
        )

    return {
        "ok": True,
        "prob": prob,
        "instability": instability,
        "mean_raw_reads": mu_raw,
        "mean_cov_log": float(mean_cov_log),
        "mean_cov_divisor": float(divisor),
        "mfe_raw": float(mfe_raw),
        "mfe_norm": float(mfe_norm),
        "reads_len": len(reads),
        "model_dim": dim,
        "pos_prob": pos_prob,
        "mature_prob": mature_prob,
        "checkpoint": state["checkpoint"],
        "arch": state.get("arch"),
    }
