#!/usr/bin/env python3
"""
核验 triplet 负样本 FASTA：upstream/downstream 是否为 stem 在基因组上紧邻的前后 padding。

约定（与本仓库 extract_padding_sequences.py 一致）：
- 正链：padding 与参考基因组 stem 紧邻区段正向一致。
- InvCom：padding 为该区段的「反向互补」full reverse-complement（与 hairpin 的 original「仅互补」不同）。

stem：header 中 stem-a-b，1-based 闭区间 [a,b]。
上游（坐标更小）：chr[max(0, a-1-pad) : a-1)。
下游：chr[b : min(len, b+pad))，b 为 1-based stem 末位，下游从 0-based 索引 b 开始。
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

HEADER_RE = re.compile(
    r"^>(\d+|X|Y|MT)_(\d+)-(\d+)(?:_(InvCom))?_stem-(\d+)-(\d+)\|(\S+)"
)
PAD_RE = re.compile(r"padding_(\d+)bp")

COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def dna_comp(s: str) -> str:
    return s.translate(COMP)


def dna_revcomp(s: str) -> str:
    return dna_comp(s[::-1])


def load_genome(path: Path, *, progress: bool = False) -> dict[str, str]:
    genome: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if cur is not None:
                    genome[cur] = "".join(buf).upper()
                    if progress:
                        print(
                            f"[load_genome] 已读 {cur}: {len(genome[cur]):,} bp",
                            file=sys.stderr,
                            flush=True,
                        )
                m = re.search(r">(\d+|X|Y|MT|chr\d+|chrX|chrY|chrMT)", line, re.I)
                if not m:
                    cur = None
                    buf = []
                    continue
                c = m.group(1)
                if c.lower().startswith("chr"):
                    c = c[3:]
                if c.upper() == "MT":
                    c = "MT"
                cur = c
                buf = []
            elif cur and line:
                buf.append(line.upper())
        if cur is not None:
            genome[cur] = "".join(buf).upper()
            if progress:
                print(
                    f"[load_genome] 已读 {cur}: {len(genome[cur]):,} bp",
                    file=sys.stderr,
                    flush=True,
                )
    return genome


def norm_seq(s: str) -> str:
    return "".join(s.split()).upper().replace("U", "T")


def part_key_from_suffix(suf: str) -> str | None:
    if suf == "original":
        return "original"
    if "upstream" in suf:
        return "upstream"
    if "downstream" in suf:
        return "downstream"
    return None


def infer_padding_size(header_line_fragment: str, default: int) -> int:
    m = PAD_RE.search(header_line_fragment)
    return int(m.group(1)) if m else default


def expected_padding(
    ch: str,
    stem_s: int,
    stem_e: int,
    pad: int,
    inv: bool,
) -> tuple[str, str]:
    up_start0 = max(0, stem_s - 1 - pad)
    up_end_excl = stem_s - 1
    up_raw = ch[up_start0:up_end_excl] if up_end_excl > up_start0 else ""

    dn_start0 = stem_e
    dn_end_excl = min(len(ch), stem_e + pad)
    dn_raw = ch[dn_start0:dn_end_excl] if dn_end_excl > dn_start0 else ""

    if inv:
        return dna_revcomp(up_raw), dna_revcomp(dn_raw)
    return up_raw, dn_raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta", type=Path)
    ap.add_argument("genome", type=Path)
    ap.add_argument("--pad", type=int, default=0, help="0 则从 header 推断 *_NNNbp")
    args = ap.parse_args()

    genome = load_genome(args.genome)
    print(f"Genome: {len(genome)} chromosomes", file=sys.stderr)

    RecordKey = tuple[str, int, int, bool]
    cur_key: RecordKey | None = None
    cur_parts: dict[str, list[str]] = defaultdict(list)
    active: str | None = None
    default_pad = 100

    n_ok = n_fail = n_skip = 0
    bad_samples: list[tuple] = []

    def process_record(key: RecordKey, parts: dict[str, list[str]], pad_len: int) -> None:
        nonlocal n_ok, n_fail, n_skip, bad_samples
        chrom, stem_s, stem_e, inv = key
        if chrom not in genome:
            n_skip += 1
            return
        ch = genome[chrom]
        up_e, dn_e = expected_padding(ch, stem_s, stem_e, pad_len, inv)
        up_f = norm_seq("".join(parts.get("upstream", [])))
        dn_f = norm_seq("".join(parts.get("downstream", [])))
        if not up_f and not dn_f:
            n_skip += 1
            return
        ok_u = up_f == up_e
        ok_d = dn_f == dn_e
        if ok_u and ok_d:
            n_ok += 1
            return
        n_fail += 1
        if len(bad_samples) < 8:
            bad_samples.append(
                (
                    key,
                    pad_len,
                    ok_u,
                    ok_d,
                    len(up_f),
                    len(up_e),
                    len(dn_f),
                    len(dn_e),
                    up_f[:40],
                    up_e[:40],
                    dn_f[:40],
                    dn_e[:40],
                )
            )

    with args.fasta.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                m = HEADER_RE.match(line)
                if not m:
                    continue
                chrom, _b0, _b1, inv, sa, se, suf = m.groups()
                invcom = inv is not None
                ss, se = int(sa), int(se)
                pk = part_key_from_suffix(suf)
                if pk is None:
                    continue
                new_key: RecordKey = (chrom, ss, se, invcom)
                if cur_key is not None and new_key != cur_key:
                    pad_guess = (
                        args.pad
                        if args.pad > 0
                        else infer_padding_size(line, default_pad)
                    )
                    process_record(cur_key, cur_parts, pad_guess)
                    cur_parts = defaultdict(list)
                cur_key = new_key
                if args.pad == 0:
                    default_pad = infer_padding_size(suf, default_pad)
                active = pk
                cur_parts[pk].clear()
            else:
                if active and cur_key is not None:
                    cur_parts[active].append(line)

    if cur_key is not None:
        pad_final = args.pad if args.pad > 0 else default_pad
        process_record(cur_key, cur_parts, pad_final)

    print("=== stem 两侧 padding 与基因组核验 ===")
    print(f"上下游均匹配记录数: {n_ok}")
    print(f"至少一侧不匹配的记录数: {n_fail}")
    print(f"跳过: {n_skip}")
    if bad_samples:
        print("\n示例不符:")
        for row in bad_samples:
            print(row)


if __name__ == "__main__":
    main()
