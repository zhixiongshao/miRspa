#!/usr/bin/env python3
"""
从 padding FASTA 逐条编码并预测，不写完整 JSON。

两种输入：

1) 三段式（与 preprocess_padding_dataset.read_fasta_with_padding 一致）：
   >seq_id|original / |upstream_padding_NNbp / |downstream_padding_NNbp

2) 仅伪发卡 RNA（无侧翼），header 须含 stem 坐标，例如：
   >7_115272450-115272610_stem-115272472-115272530
   >5_134171880-134172040_InvCom_stem-134171958-134172033
   此时须指定 --genome：按 verify_stem_padding_vs_genome.expected_padding 在 stem 两侧
   各取 padding_size bp（正链取基因组原序；InvCom 为两侧片段的反向互补），再按
   reverse_original_swap_padding 约定交换上下游内容（与 …_reversed_swap_ud 训练数据一致），
   前体序列使用 FASTA 正文（已是 reversed 后的 RNA）。

默认输入/输出（可用参数覆盖）：
  FASTA: github/true_pseudo_hairpin_sampled_1000_precise_stem_invcom_reversed.fasta
  TSV:   同目录 true_pseudo_hairpin_sampled_1000_precise_stem_invcom_reversed_predictions_stream.tsv

用法示例：
  python3 stream_predict_padding_fasta.py --genome /path/to/GRCh38.primary_assembly.fa
  python3 stream_predict_padding_fasta.py --fasta other.fa --out other.tsv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch

# 与 JSON 管线完全相同的编码与折叠逻辑
import RNA

from preprocess_padding_dataset import (
    STRUCT_CHANNEL_ABSENT,
    build_full_encoded_sequence,
    dna_to_rna,
)
from occlusion_saliency_mlp import load_model, positive_class_prob
from verify_stem_padding_vs_genome import expected_padding, load_genome

DEFAULT_FASTA = (
    "/mnt/data/home/zhixiong/MiRNA/github/"
    "true_pseudo_hairpin_sampled_1000_precise_stem_invcom_reversed.fasta"
)
DEFAULT_OUT_TSV = (
    "/mnt/data/home/zhixiong/MiRNA/github/"
    "true_pseudo_hairpin_sampled_1000_precise_stem_invcom_reversed_predictions_stream.tsv"
)

# 无 |original 后缀的伪发卡 FASTA：chr_bin_InvCom?_stem-a-b
HAIRPIN_STEM_HEADER_RE = re.compile(
    r"^(\d+|X|Y|MT)_(\d+)-(\d+)(?:_(InvCom))?_stem-(\d+)-(\d+)$",
    re.IGNORECASE,
)


def parse_hairpin_stem_header(seq_id: str) -> tuple[str, int, int, bool] | None:
    """
    解析 chr_bin[_InvCom]_stem-a-b。
    若 stem 坐标小于 bin 起点，视为窗口内 1-based 相对坐标，转为基因组绝对坐标：
    abs = bin_start + rel - 1（与 header 中 bin 第一段数字一致）。
    """
    m = HAIRPIN_STEM_HEADER_RE.match(seq_id.strip())
    if not m:
        return None
    chrom = m.group(1)
    if chrom.upper() == "MT":
        chrom = "MT"
    bin_a, _bin_b = int(m.group(2)), int(m.group(3))
    inv = m.group(4) is not None
    stem_s, stem_e = int(m.group(5)), int(m.group(6))
    if stem_s < bin_a:
        stem_s = bin_a + stem_s - 1
        stem_e = bin_a + stem_e - 1
    return chrom, stem_s, stem_e, inv


def attach_padding_from_genome(
    seq_id: str,
    seq_data: dict[str, str],
    genome: dict[str, str],
    padding_size: int,
    *,
    require_stem: bool,
) -> tuple[dict[str, str], str | None]:
    """
    若记录无上下游且 header 含 stem，则从基因组取 padding 并做 reversed_swap 交换。
    返回 (新 seq_data, 错误信息)；无错误时第二项为 None。
    require_stem：为 True 时（已传 --genome）无侧翼则必须能解析 stem，否则返回错误。
    """
    if seq_data.get("upstream") or seq_data.get("downstream"):
        return seq_data, None
    parsed = parse_hairpin_stem_header(seq_id)
    if parsed is None:
        if require_stem:
            return seq_data, "no_stem_header"
        return seq_data, None
    chrom, stem_s, stem_e, inv = parsed
    ch = genome.get(chrom)
    if ch is None:
        return seq_data, f"no_chromosome:{chrom}"
    up_dna, dn_dna = expected_padding(ch, stem_s, stem_e, padding_size, inv)
    # 与 reverse_original_swap_padding 一致：写入文件的 upstream=原下游，downstream=原上游
    merged = {
        "original": seq_data["original"],
        "upstream": dn_dna,
        "downstream": up_dna,
    }
    return merged, None


def _parse_fasta_header(rest: str) -> tuple[str | None, str | None]:
    """与 preprocess_padding_dataset.read_fasta_with_padding 头部解析一致。"""
    if (
        "|original" in rest
        or "|upstream_padding_" in rest
        or "|downstream_padding_" in rest
    ):
        if "|" not in rest:
            return None, None
        seq_id, seq_type = rest.split("|", 1)
        seq_id = seq_id.strip()
        if "original" in seq_type:
            return seq_id, "original"
        if "upstream_padding" in seq_type:
            return seq_id, "upstream"
        if "downstream_padding" in seq_type:
            return seq_id, "downstream"
        return None, None
    seq_id = rest.strip()
    return (seq_id, "original") if seq_id else (None, None)


def iter_padding_fasta_records(fasta_path: str):
    """
    流式读取 FASTA，每读完一条序列（同一 seq_id 的各段）即 yield (seq_id, parts_dict)。
    parts 含键 original，以及可选 upstream、downstream。
    """
    active_id: str | None = None
    cur_type: str | None = None
    buf: list[str] = []
    parts: dict[str, str] = {}

    def flush_buf() -> None:
        nonlocal buf
        if active_id is not None and cur_type is not None and buf:
            parts[cur_type] = "".join(buf).upper()
        buf = []

    with open(fasta_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_buf()
                new_id, new_type = _parse_fasta_header(line[1:].strip())
                if new_id is None:
                    active_id, cur_type = None, None
                    continue
                if active_id is not None and new_id != active_id:
                    if "original" in parts:
                        yield active_id, dict(parts)
                    parts = {}
                active_id = new_id
                cur_type = new_type
            else:
                if active_id is not None and cur_type is not None:
                    buf.append(line)

        flush_buf()
        if active_id is not None and "original" in parts:
            yield active_id, dict(parts)


def encode_like_preprocess_positive(
    seq_data: dict[str, str],
    *,
    padding_size: int,
    premirna_len: int,
    struct_absent: float,
) -> tuple[np.ndarray, float] | None:
    """
    与 preprocess_padding_dataset.process_padding_dataset 正样本分支相同
    （成功返回 (5, L) float32 与 mfe；失败返回 None）。
    """
    if "original" not in seq_data:
        return None

    original_seq = seq_data["original"]
    upstream_seq = seq_data.get("upstream", "")
    downstream_seq = seq_data.get("downstream", "")

    original_rna = dna_to_rna(original_seq)
    if "N" in original_rna.upper():
        return None

    try:
        structure, mfe = RNA.fold(original_rna)
    except Exception:
        return None

    if len(structure) != len(original_rna):
        return None

    upstream_rna = dna_to_rna(upstream_seq) if upstream_seq else ""
    downstream_rna = dna_to_rna(downstream_seq) if downstream_seq else ""

    total_len = padding_size + premirna_len + padding_size
    full_encoded = build_full_encoded_sequence(
        upstream_rna=upstream_rna,
        premirna_rna=original_rna,
        premirna_structure=structure,
        downstream_rna=downstream_rna,
        target_total_len=total_len,
        struct_absent=struct_absent,
    )
    return full_encoded.astype(np.float32), float(mfe)


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    encoded: np.ndarray,
    mfe: float,
    *,
    device: torch.device,
    use_mfe: bool,
    max_seq_len: int,
) -> float:
    """单条前向，正类概率（与 test 中二分类一致）。"""
    x = torch.from_numpy(encoded).unsqueeze(0).to(device)
    if use_mfe:
        mfe_np = np.array([mfe], dtype=np.float32)
        seq_len_np = np.array([float(max_seq_len)], dtype=np.float32)
        out = model(x, mfe=mfe_np, seq_len=seq_len_np)
    else:
        out = model(x)
    p = positive_class_prob(out, model)
    if p.dim() > 0:
        p = p.reshape(-1)[0]
    return float(p.item())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="三段式 padding FASTA 流式编码 + 预测（不写 JSON）"
    )
    ap.add_argument(
        "--fasta",
        type=str,
        default=DEFAULT_FASTA,
        help=f"输入 FASTA（默认: {DEFAULT_FASTA}）",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/data/home/zhixiong/MiRNA/github/dataset/miRspa1000/best_model.pth",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT_TSV,
        help=f"输出 TSV（默认: {DEFAULT_OUT_TSV}）",
    )
    ap.add_argument(
        "--padding_size",
        type=int,
        default=1000,
        help="与生成 FASTA 时侧翼长度一致（默认 1000）",
    )
    ap.add_argument(
        "--genome",
        type=str,
        default=None,
        help="GRCh38 等整基因组 FASTA；伪发卡-only FASTA 且无侧翼时必填，用于按 stem 坐标取 padding",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--struct_absent",
        type=float,
        default=STRUCT_CHANNEL_ABSENT,
        help="与 preprocess_padding_dataset 默认一致",
    )
    args = ap.parse_args()

    fasta_path = Path(args.fasta)
    if not fasta_path.is_file():
        print(f"错误: 找不到 FASTA: {fasta_path}", file=sys.stderr)
        sys.exit(1)

    ckpt_path = Path(args.checkpoint)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, model_config = load_model(str(ckpt_path), dev)
    use_mfe = model_config.get("use_mfe", False)
    max_seq_len = int(model_config["max_seq_len"])
    premirna_len = max_seq_len - 2 * args.padding_size
    if premirna_len < 1:
        print(
            f"错误: max_seq_len={max_seq_len} 与 padding_size={args.padding_size} 不兼容",
            file=sys.stderr,
        )
        sys.exit(1)

    genome_dict: dict[str, str] | None = None
    if args.genome:
        gp = Path(args.genome)
        if not gp.is_file():
            print(f"错误: 找不到基因组 FASTA: {gp}", file=sys.stderr)
            sys.exit(1)
        print("加载基因组（用于按 stem 取 padding）…", file=sys.stderr)
        genome_dict = load_genome(gp)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = 0
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout, delimiter="\t")
        w.writerow(["seq_id", "prob_positive", "status"])
        fout.flush()

        for seq_id, seq_data in iter_padding_fasta_records(str(fasta_path)):
            if genome_dict is not None:
                seq_data, pad_err = attach_padding_from_genome(
                    seq_id,
                    seq_data,
                    genome_dict,
                    args.padding_size,
                    require_stem=True,
                )
                if pad_err:
                    w.writerow([seq_id, "", f"skip:{pad_err}"])
                    n_skip += 1
                    fout.flush()
                    continue
            enc = encode_like_preprocess_positive(
                seq_data,
                padding_size=args.padding_size,
                premirna_len=premirna_len,
                struct_absent=args.struct_absent,
            )
            if enc is None:
                w.writerow([seq_id, "", "skip_encode"])
                n_skip += 1
                fout.flush()
                continue

            encoded, mfe = enc
            try:
                prob = predict_one(
                    model,
                    encoded,
                    mfe,
                    device=dev,
                    use_mfe=use_mfe,
                    max_seq_len=max_seq_len,
                )
            except Exception as e:
                w.writerow([seq_id, "", f"error:{e!s}"])
                n_skip += 1
                fout.flush()
                continue

            w.writerow([seq_id, f"{prob:.8f}", "ok"])
            n_ok += 1
            fout.flush()

    print(f"完成: ok={n_ok}, skip/error={n_skip}, 输出: {out_path}")


if __name__ == "__main__":
    main()
