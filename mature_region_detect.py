"""从 mature_prob / pos_prob profile 检测 5p/3p 成熟体区间。

逻辑与 ``Pancancer/export_pancancer1142_mature_5p_3p_fasta.py`` 一致：
在有效茎区 profile 内结合 pos_prob 活性区与自适应 mature_prob 阈值找 1–2 段高峰；
按 profile 5'→3'（index 0 = stem_start 侧）将靠左峰标为 5p、靠右峰标为 3p。
"""
from __future__ import annotations

from typing import Any

import numpy as np

DIM = 180
MIN_MATURE_LEN = 12
MAX_MATURE_LEN = 30
IDEAL_MATURE_LEN = 22
GAP_MERGE = 4


def _smooth3(x: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return x.copy()
    y = x.copy()
    y[1:-1] = (x[:-2] + x[1:-1] + x[2:]) / 3.0
    return y


def _connected_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    i, n = 0, int(mask.size)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def _merge_runs(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    runs = sorted(runs)
    out = [runs[0]]
    for a0, a1 in runs[1:]:
        p0, p1 = out[-1]
        if a0 - p1 - 1 <= gap:
            out[-1] = (p0, max(p1, a1))
        else:
            out.append((a0, a1))
    return out


def _best_window(mat: np.ndarray, i0: int, i1: int) -> tuple[int, int]:
    """在 [i0,i1] 内取长度 ∈ [MIN,MAX] 且 mature_prob 和最大的窗口。"""
    lo = max(0, i0)
    hi = min(int(mat.size) - 1, i1)
    span = hi - lo + 1
    if span <= MAX_MATURE_LEN:
        return lo, hi
    best_s, best_a, best_b = -1.0, lo, lo + IDEAL_MATURE_LEN - 1
    for w in range(MIN_MATURE_LEN, MAX_MATURE_LEN + 1):
        if w > span:
            break
        csum = np.concatenate([[0.0], np.cumsum(mat[lo : hi + 1], dtype=np.float64)])
        for a in range(0, span - w + 1):
            s = float(csum[a + w] - csum[a])
            if s > best_s:
                best_s, best_a, best_b = s, lo + a, lo + a + w - 1
    return best_a, best_b


def _shrink_run(mat: np.ndarray, i0: int, i1: int) -> tuple[int, int]:
    length = i1 - i0 + 1
    if length <= MAX_MATURE_LEN:
        if length < MIN_MATURE_LEN and i1 + (MIN_MATURE_LEN - length) < mat.size:
            return i0, min(mat.size - 1, i0 + MIN_MATURE_LEN - 1)
        return i0, i1
    return _best_window(mat, i0, i1)


def find_mature_regions(
    mature: np.ndarray,
    pos: np.ndarray,
    raw_len: int,
) -> list[dict[str, Any]]:
    """
    返回列表，每项: arm in {5p,3p}, i0, i1 (0-based profile), mean_prob, peak_prob。
    """
    L = min(int(raw_len), DIM, int(mature.size), int(pos.size))
    if L < MIN_MATURE_LEN:
        return []

    m = _smooth3(mature[:L].astype(np.float64))
    p = pos[:L].astype(np.float64)

    active = p >= 0.15
    if int(active.sum()) < 8:
        active = np.ones(L, dtype=bool)

    m_act = np.where(active, m, 0.0)
    peak = float(m_act.max())
    if peak < 0.08:
        return []

    thr = max(0.10, 0.38 * peak, peak - 0.18)
    mask = (m_act >= thr) & active
    runs = _merge_runs(_connected_runs(mask), GAP_MERGE)

    kept: list[tuple[int, int, float, float]] = []
    for a0, a1 in runs:
        if a1 - a0 + 1 < 8:
            continue
        sub = m[a0 : a1 + 1]
        if float(sub.mean()) < thr * 0.55:
            continue
        a0, a1 = _shrink_run(m, a0, a1)
        kept.append((a0, a1, float(sub.mean()), float(sub.max())))

    if not kept:
        j = int(np.argmax(m_act))
        a0 = max(0, j - IDEAL_MATURE_LEN // 2)
        a1 = min(L - 1, a0 + IDEAL_MATURE_LEN - 1)
        a0, a1 = _shrink_run(m, a0, a1)
        sub = m[a0 : a1 + 1]
        kept = [(a0, a1, float(sub.mean()), float(sub.max()))]

    if len(kept) > 2:
        scores = [(a1 - a0 + 1) * mean_p for a0, a1, mean_p, _ in kept]
        order = sorted(range(len(kept)), key=lambda i: -scores[i])[:2]
        kept = [kept[i] for i in sorted(order)]

    active_idx = np.where(active)[0]
    a_left, a_right = int(active_idx[0]), int(active_idx[-1])
    mid_active = 0.5 * (a_left + a_right)

    out: list[dict[str, Any]] = []
    if len(kept) == 1:
        a0, a1, mean_p, max_p = kept[0]
        mid = 0.5 * (a0 + a1)
        arm = "5p" if mid <= mid_active else "3p"
        out.append({"arm": arm, "i0": a0, "i1": a1, "mean_prob": mean_p, "peak_prob": max_p})
    else:
        kept = sorted(kept, key=lambda x: x[0])
        arms = ["5p", "3p"]
        for arm, (a0, a1, mean_p, max_p) in zip(arms, kept):
            out.append({"arm": arm, "i0": a0, "i1": a1, "mean_prob": mean_p, "peak_prob": max_p})
    return out


def profile_index_to_genomic_1b(
    idx: int,
    *,
    stem_start_1b: int,
    stem_end_1b: int,
    strand: str,
) -> int:
    """profile 下标 → 基因组 1-based 坐标（与 stem 序列方向一致）。"""
    if str(strand).strip() == "-":
        return int(stem_end_1b) - int(idx)
    return int(stem_start_1b) + int(idx)


def mature_regions_to_genomic(
    regions: list[dict[str, Any]],
    *,
    stem_start_1b: int,
    stem_end_1b: int,
    strand: str,
    chrom: str,
    stem_rna: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for reg in regions:
        i0 = int(reg["i0"])
        i1 = int(reg["i1"])
        g0 = profile_index_to_genomic_1b(
            i0, stem_start_1b=stem_start_1b, stem_end_1b=stem_end_1b, strand=strand
        )
        g1 = profile_index_to_genomic_1b(
            i1, stem_start_1b=stem_start_1b, stem_end_1b=stem_end_1b, strand=strand
        )
        seq = stem_rna[i0 : i1 + 1] if stem_rna else ""
        out.append(
            {
                "arm": str(reg["arm"]),
                "stem_start_idx": i0,
                "stem_end_idx": i1,
                "genomic_start_1b": min(g0, g1),
                "genomic_end_1b": max(g0, g1),
                "chrom": chrom,
                "strand": strand,
                "sequence_rna": seq,
                "mean_prob": float(reg["mean_prob"]),
                "peak_prob": float(reg["peak_prob"]),
            }
        )
    return out
