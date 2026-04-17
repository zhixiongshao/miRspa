#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
import torch


def _find_mirna_repo_root() -> Path:
    """
    在父目录链上查找含 occlusion_saliency_mlp.py 的 MiRNA 项目根。
    避免 Render 上 Root Directory / 部署路径与本地 parents[2] 不一致导致 import 失败。
    """
    start = Path(__file__).resolve().parent
    for d in [start, *start.parents]:
        if (d / "occlusion_saliency_mlp.py").is_file():
            return d
    raise SystemExit(
        "找不到 occlusion_saliency_mlp.py：请确认 Git 推送的是完整 MiRNA 仓库（根目录须有该文件），\n"
        "且 Render 的 Root Directory 设为仓库根（不要只部署 github/web 子目录）。\n"
        f"app.py 位置: {Path(__file__).resolve()}"
    )


ROOT = _find_mirna_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from occlusion_saliency_mlp import forward_batch, load_model
from preprocess_padding_dataset import STRUCT_CHANNEL_ABSENT
from stream_predict_padding_fasta import (
    encode_like_preprocess_positive,
    parse_hairpin_stem_header,
)
from verify_stem_padding_vs_genome import expected_padding


DEFAULT_CHECKPOINT = ROOT / "github/dataset/miRspa1000/best_model.pth"
DEFAULT_GENOME = ROOT / "gene/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
INDEX_HTML = Path(__file__).with_name("index.html")
WEB_DIR = Path(__file__).resolve().parent
"""
Heatmap feature removed per request.
"""


R2DT_BASE_DEFAULT = "https://www.ebi.ac.uk/Tools/services/rest/r2dt"

_R2DT_JOB_LOCK = threading.Lock()
_R2DT_INFLIGHT: dict[str, dict[str, object]] = {}


def _http_request(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 60.0,
) -> tuple[int, bytes, dict[str, str]]:
    hdrs = {
        "User-Agent": "MiRNAWebPredictor/1.0 (+https://github.com/)",
        "Accept": "*/*",
    }
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, method=method, headers=hdrs)
    with urlopen(req, timeout=timeout_s) as resp:  # nosec - controlled URL
        status = int(getattr(resp, "status", 200))
        body = resp.read()
        h = {k.lower(): v for k, v in resp.headers.items()}
        return status, body, h


def _http_request_text(url: str, *, timeout_s: float = 60.0) -> str:
    _st, body, _h = _http_request(url, timeout_s=timeout_s)
    return body.decode("utf-8", errors="replace").strip()


def _r2dt_submit_job(
    *,
    base: str,
    email: str,
    template_id: str,
    fold_type: str,
    constraint: str,
    fasta: str,
    timeout_s: float = 120.0,
) -> str:
    url = f"{base.rstrip('/')}/run"
    form = {
        "email": email,
        "template_id": template_id,
        "fold_type": fold_type,
        "constraint": constraint,
        "sequence": fasta,
    }
    body = urlencode(form, doseq=False).encode("utf-8")
    _st, resp, _h = _http_request(
        url,
        method="POST",
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
        timeout_s=timeout_s,
    )
    job_id = resp.decode("utf-8", errors="replace").strip()
    if not job_id or "\n" in job_id or " " in job_id:
        raise RuntimeError(f"R2DT run 返回异常 job id: {job_id!r}")
    return job_id


def _r2dt_wait_finished(
    *,
    base: str,
    job_id: str,
    poll_s: float,
    max_wait_s: float,
) -> None:
    url = f"{base.rstrip('/')}/status/{job_id}"
    deadline = time.monotonic() + float(max_wait_s)
    last = ""
    while time.monotonic() < deadline:
        try:
            st = _http_request_text(url, timeout_s=60.0)
        except HTTPError as e:
            # 过期/不存在时常见 404；交给上层触发重新提交。
            if int(getattr(e, "code", 0) or 0) in {404, 410}:
                raise
            raise
        last = st
        if st in {"FINISHED", "DONE", "COMPLETED"}:
            return
        if st in {"FAILED", "ERROR", "NOT_AVAILABLE", "FAILURE"}:
            raise RuntimeError(f"R2DT 任务失败: job={job_id} status={st!r}")
        time.sleep(float(poll_s))
    raise TimeoutError(f"R2DT 任务超时: job={job_id} last_status={last!r}")


def _r2dt_fetch_svg(*, base: str, job_id: str, timeout_s: float = 120.0) -> bytes:
    url = f"{base.rstrip('/')}/result/{job_id}/svg"
    _st, body, _h = _http_request(url, timeout_s=timeout_s)
    if not body:
        raise RuntimeError("R2DT svg 结果为空")
    return body


def render_hairpin_r2dt_svg(
    *,
    seq: str,
    structure: str,
    seq_id: str,
) -> tuple[str, str | None]:
    """
    通过 EBI R2DT REST 生成 SVG。

    重要说明：
    - EBI 的 job id **不是永久资源**（通常会在一段时间后过期），因此不要把它当作长期缓存 key。
    - 这里用 (序列 + 二级结构 + 参数) 的 hash 做本地文件缓存；过期/404 会自动重新 run。

    返回：(svg 或空字符串, error_message_or_None)
    """
    if os.environ.get("R2DT_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "", "R2DT_DISABLE=1"

    email = (os.environ.get("R2DT_EMAIL") or os.environ.get("EBI_TOOLS_EMAIL") or "").strip()
    if not email:
        return "", "缺少环境变量 R2DT_EMAIL（或 EBI_TOOLS_EMAIL）"

    base = (os.environ.get("R2DT_BASE") or R2DT_BASE_DEFAULT).strip()
    template_id = (os.environ.get("R2DT_TEMPLATE_ID") or "auto").strip() or "auto"
    fold_type = (os.environ.get("R2DT_FOLD_TYPE") or "constraint").strip() or "constraint"
    constraint = (os.environ.get("R2DT_CONSTRAINT") or structure).strip()

    poll_s = float(os.environ.get("R2DT_POLL_S", "2.0") or "2.0")
    max_wait_s = float(os.environ.get("R2DT_MAX_WAIT_S", "900") or "900")
    max_attempts = int(os.environ.get("R2DT_MAX_ATTEMPTS", "3") or "3")

    cache_dir = Path(os.environ.get("R2DT_CACHE_DIR") or (WEB_DIR / ".r2dt_cache")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    seq_norm = "".join(str(seq).split()).upper().replace("T", "U")
    struct_norm = "".join(str(structure).split())
    if not seq_norm or not struct_norm:
        return "", "序列为空或二级结构为空，无法调用 R2DT"

    n = min(len(seq_norm), len(struct_norm))
    seq_norm = seq_norm[:n]
    struct_norm = struct_norm[:n]

    key_src = "|".join(
        [
            "v1",
            base,
            template_id,
            fold_type,
            constraint,
            seq_norm,
            struct_norm,
        ]
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{key}.svg"

    if cache_path.is_file() and cache_path.stat().st_size > 0:
        return cache_path.read_text(encoding="utf-8", errors="replace"), None

    sid = re.sub(r"[^A-Za-z0-9_.-]+", "_", (seq_id or "hairpin").strip())[:80] or "hairpin"
    fasta = f">{sid}\n{seq_norm}\n{struct_norm}\n"

    last_err: str | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        try:
            job_id = _r2dt_submit_job(
                base=base,
                email=email,
                template_id=template_id,
                fold_type=fold_type,
                constraint=constraint,
                fasta=fasta,
            )
            _r2dt_wait_finished(
                base=base,
                job_id=job_id,
                poll_s=poll_s,
                max_wait_s=max_wait_s,
            )
            svg_bytes = _r2dt_fetch_svg(base=base, job_id=job_id)
            svg = svg_bytes.decode("utf-8", errors="replace")
            if "<svg" not in svg.lower():
                raise RuntimeError("下载内容不像 SVG")
            tmp = cache_path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(svg_bytes)
            tmp.replace(cache_path)
            return svg, None
        except HTTPError as e:
            code = int(getattr(e, "code", 0) or 0)
            last_err = f"HTTP {code}: {e}"
            # 常见：job 过期/不存在 -> 重新提交
            if code in {404, 410, 500, 502, 503, 504}:
                time.sleep(min(5.0, 0.5 * attempt))
                continue
            return "", last_err
        except (URLError, TimeoutError, RuntimeError, OSError) as e:
            last_err = str(e)
            time.sleep(min(5.0, 0.5 * attempt))
            continue

    return "", last_err or "R2DT 生成失败"


def _compute_r2dt_cache_key_and_path(*, seq: str, structure: str) -> tuple[str, Path, str | None]:
    if os.environ.get("R2DT_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "", Path("."), "R2DT_DISABLE=1"

    email = (os.environ.get("R2DT_EMAIL") or os.environ.get("EBI_TOOLS_EMAIL") or "").strip()
    if not email:
        return "", Path("."), "缺少环境变量 R2DT_EMAIL（或 EBI_TOOLS_EMAIL）"

    base = (os.environ.get("R2DT_BASE") or R2DT_BASE_DEFAULT).strip()
    template_id = (os.environ.get("R2DT_TEMPLATE_ID") or "auto").strip() or "auto"
    fold_type = (os.environ.get("R2DT_FOLD_TYPE") or "constraint").strip() or "constraint"
    constraint = (os.environ.get("R2DT_CONSTRAINT") or structure).strip()

    cache_dir = Path(os.environ.get("R2DT_CACHE_DIR") or (WEB_DIR / ".r2dt_cache")).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    seq_norm = "".join(str(seq).split()).upper().replace("T", "U")
    struct_norm = "".join(str(structure).split())
    if not seq_norm or not struct_norm:
        return "", Path("."), "序列为空或二级结构为空，无法调用 R2DT"

    n = min(len(seq_norm), len(struct_norm))
    seq_norm = seq_norm[:n]
    struct_norm = struct_norm[:n]

    key_src = "|".join(
        [
            "v1",
            base,
            template_id,
            fold_type,
            constraint,
            seq_norm,
            struct_norm,
        ]
    )
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    return key, (cache_dir / f"{key}.svg"), None


def _enqueue_r2dt_svg_generation(*, key: str, seq: str, structure: str, seq_id: str) -> None:
    # Avoid duplicate threads.
    with _R2DT_JOB_LOCK:
        info = _R2DT_INFLIGHT.get(key)
        if info and info.get("status") in {"running", "done"}:
            return
        _R2DT_INFLIGHT[key] = {"status": "running", "error": None, "started_at": time.time()}

    def worker() -> None:
        try:
            svg, err = render_hairpin_r2dt_svg(seq=seq, structure=structure, seq_id=seq_id)
            with _R2DT_JOB_LOCK:
                _R2DT_INFLIGHT[key] = {"status": "done", "error": err, "finished_at": time.time()}
        except BaseException as e:
            with _R2DT_JOB_LOCK:
                _R2DT_INFLIGHT[key] = {"status": "done", "error": str(e), "finished_at": time.time()}

    threading.Thread(target=worker, daemon=True, name=f"r2dt-{key[:10]}").start()


def get_hairpin_svg_or_enqueue(
    *,
    seq: str,
    structure: str,
    seq_id: str,
) -> dict[str, object]:
    """
    非阻塞：优先读本地缓存，未命中则后台生成，并返回 pending+key 供前端轮询。
    """
    key, cache_path, err = _compute_r2dt_cache_key_and_path(seq=seq, structure=structure)
    if err is not None:
        return {
            "hairpin_key": None,
            "hairpin_pending": False,
            "hairpin_svg": "",
            "hairpin_svg_error": err,
        }

    if cache_path.is_file() and cache_path.stat().st_size > 0:
        svg = cache_path.read_text(encoding="utf-8", errors="replace")
        return {
            "hairpin_key": key,
            "hairpin_pending": False,
            "hairpin_svg": svg,
            "hairpin_svg_error": None,
        }

    _enqueue_r2dt_svg_generation(key=key, seq=seq, structure=structure, seq_id=seq_id)
    with _R2DT_JOB_LOCK:
        info = dict(_R2DT_INFLIGHT.get(key) or {})
    pending = info.get("status") != "done"
    return {
        "hairpin_key": key,
        "hairpin_pending": bool(pending),
        "hairpin_svg": "",
        "hairpin_svg_error": info.get("error"),
    }

# Zenodo record: https://zenodo.org/records/19586578
# DOI: https://doi.org/10.5281/zenodo.19586578
DEFAULT_ZENODO_CHECKPOINT_URL = (
    "https://zenodo.org/records/19586578/files/best_model.pth?download=1"
)
DEFAULT_ZENODO_GENOME_GZ_URL = (
    "https://zenodo.org/records/19586578/files/"
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz?download=1"
)

DEFAULT_SEQ_API_BASE = "https://rest.ensembl.org"

DNA_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
STATE_LOCK = threading.Lock()
APP_STATE: dict[str, object] = {
    "ready": False,
    "model": None,
    "device": None,
    "use_mfe": False,
    "padding_size": 1000,
    "premirna_len": None,
    "max_seq_len": None,
    "checkpoint": None,
    "genome_path": None,
}

# Render 会在进程启动后很快检测 $PORT；大文件下载不能阻塞 bind，否则报 “No open ports”。
_ASSETS_BOOTSTRAP_DONE = threading.Event()
_ASSETS_BOOTSTRAP_ERR: str | None = None


def _bootstrap_assets_thread_body(args: argparse.Namespace) -> None:
    global _ASSETS_BOOTSTRAP_ERR
    try:
        ensure_local_prediction_assets(args)
        if not args.checkpoint.is_file():
            raise RuntimeError(f"找不到 checkpoint: {args.checkpoint}")
        if not _use_sequence_api():
            if not _genome_asset_ready(args.genome):
                raise RuntimeError(
                    f"找不到参考基因组: {args.genome}（可为 .fa 或同路径的 .fa.gz）"
                )
            resolve_genome_source_path(args.genome)
    except BaseException as e:
        _ASSETS_BOOTSTRAP_ERR = str(e)
        print(f"[web] asset bootstrap failed: {e}", file=sys.stderr, flush=True)
    finally:
        _ASSETS_BOOTSTRAP_DONE.set()


def _assets_ready_for_predict() -> str | None:
    """若可预测返回 None；否则返回给客户端的错误说明。"""
    if not _ASSETS_BOOTSTRAP_DONE.is_set():
        return (
            "服务正在准备数据（首次启动会从 Zenodo 下载大文件），请稍后重试 /api/predict"
        )
    if _ASSETS_BOOTSTRAP_ERR is not None:
        return f"资源准备失败: {_ASSETS_BOOTSTRAP_ERR}"
    return None


# 与 verify_stem_padding_vs_genome.load_genome 的染色体解析保持一致
_HEADER_CHROM_RE = re.compile(r">(\d+|X|Y|MT|chr\d+|chrX|chrY|chrMT)", re.I)

def _use_sequence_api() -> bool:
    return os.environ.get("USE_SEQUENCE_API", "").strip().lower() in {"1", "true", "yes"}


def _zenodo_fetch_disabled() -> bool:
    return os.environ.get("DISABLE_ZENODO_ASSET_FETCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _stream_download(url: str, dest: Path, *, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    if part.is_file():
        part.unlink()
    req = Request(url, headers={"User-Agent": "MiRNA-web-predictor/1.0"})
    print(f"[web] downloading {label} -> {dest}", file=sys.stderr, flush=True)
    try:
        with urlopen(req) as resp, part.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1024 * 1024)
        part.replace(dest)
    except (URLError, OSError) as e:
        if part.is_file():
            part.unlink(missing_ok=True)
        raise SystemExit(f"下载失败 ({label}): {e}") from e


def _genome_gz_destination(genome: Path) -> Path:
    """Zenodo / disk 上的参考序列：统一落到 .fa.gz 路径。"""
    if genome.suffix.lower() == ".gz":
        return genome
    if genome.suffix.lower() == ".fa":
        return genome.with_suffix(".fa.gz")
    return genome.with_suffix(genome.suffix + ".gz")


def _genome_asset_ready(genome: Path) -> bool:
    if genome.is_file():
        return True
    if genome.suffix.lower() == ".fa":
        return _genome_gz_destination(genome).is_file()
    return False


def resolve_genome_source_path(genome: Path) -> Path:
    """
    实际打开的路径：支持 .fa 或 .fa.gz。
    若只存在同主名的 .fa.gz（用户仍传 .fa），则返回 .fa.gz。
    """
    if genome.is_file():
        return genome
    if genome.suffix.lower() == ".fa":
        gz = genome.with_suffix(".fa.gz")
        if gz.is_file():
            return gz
    raise FileNotFoundError(f"找不到参考基因组文件: {genome}")


def header_chrom_key_from_line(header_line: str) -> str | None:
    m = _HEADER_CHROM_RE.search(header_line)
    if not m:
        return None
    c = m.group(1)
    if c.lower().startswith("chr"):
        c = c[3:]
    if c.upper() == "MT":
        return "MT"
    if c.isdigit():
        return str(int(c))
    return c.upper()


def extract_chrom_window_from_fasta(
    path: Path,
    chrom_key: str,
    window_start0: int,
    window_end_excl: int,
    *,
    progress: bool = False,
) -> str:
    """
    从多染色体 FASTA / .fa.gz 中只读取一条染色体上的连续窗口
    [window_start0, window_end_excl)（0-based，半开区间），不解压整库、不加载整条染色体。
    若染色体在窗口右端之前结束，则返回较短片段（与全序列 min(len, end) 行为一致）。
    """
    if window_end_excl <= window_start0:
        raise ValueError("基因组窗口无效：window_end_excl 必须大于 window_start0")
    target = normalize_chrom(chrom_key)
    if path.suffix.lower() == ".gz":
        ctx = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        ctx = path.open("r", encoding="utf-8", errors="replace")

    in_target = False
    found = False
    idx = 0
    collected: list[str] = []

    with ctx as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if in_target:
                    break
                key = header_chrom_key_from_line(line)
                if key == target:
                    in_target = True
                    found = True
                    idx = 0
                    collected = []
                    if progress:
                        print(
                            f"[load_genome] 读取 {target} 窗口 "
                            f"[{window_start0:,}, {window_end_excl:,}) ({path.name})",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    in_target = False
                continue
            if not in_target:
                continue
            s = line.upper()
            ln = len(s)
            seg_start = idx
            seg_end = idx + ln
            if seg_end <= window_start0:
                idx = seg_end
                if idx >= window_end_excl:
                    break
                continue
            lo = max(window_start0, seg_start)
            hi = min(window_end_excl, seg_end)
            if lo < hi:
                collected.append(s[lo - seg_start : hi - seg_start])
            idx = seg_end
            if idx >= window_end_excl:
                break

    if not found:
        raise ValueError(f"在参考序列文件中找不到染色体: {target} ({path})")
    out = "".join(collected)
    if not out:
        raise ValueError(f"染色体 {target} 窗口序列为空")
    if progress:
        print(
            f"[load_genome] 窗口实际长度 {len(out):,} bp（请求 "
            f"{window_end_excl - window_start0:,} bp）",
            file=sys.stderr,
            flush=True,
        )
    return out


def attach_padding_with_window(
    seq_id: str,
    seq_data: dict[str, str],
    window: str,
    window_start0: int,
    padding_size: int,
    *,
    require_stem: bool,
) -> tuple[dict[str, str], str | None]:
    """与 attach_padding_from_genome 相同语义，但 reference 仅为局部窗口字符串。"""
    if seq_data.get("upstream") or seq_data.get("downstream"):
        return seq_data, None
    parsed = parse_hairpin_stem_header(seq_id)
    if parsed is None:
        if require_stem:
            return seq_data, "no_stem_header"
        return seq_data, None
    _chrom, stem_s, stem_e, inv = parsed
    stem_s_loc = stem_s - window_start0
    stem_e_loc = stem_e - window_start0
    if stem_s_loc < 1 or stem_e_loc < stem_s_loc or stem_e_loc > len(window):
        return seq_data, "stem_outside_window"
    up_dna, dn_dna = expected_padding(window, stem_s_loc, stem_e_loc, padding_size, inv)
    merged = {
        "original": seq_data["original"],
        "upstream": dn_dna,
        "downstream": up_dna,
    }
    return merged, None


def ensure_local_prediction_assets(args: argparse.Namespace) -> None:
    """
    若 checkpoint / 基因组 .fa.gz 缺失，则从 Zenodo 拉取（不整库解压 .fa）。
    可用 DISABLE_ZENODO_ASSET_FETCH=1 关闭；URL 用 CHECKPOINT_URL / GENOME_GZ_URL 覆盖。
    """
    if _zenodo_fetch_disabled():
        return

    ckpt_url = os.environ.get("CHECKPOINT_URL", DEFAULT_ZENODO_CHECKPOINT_URL).strip()
    gz_url = os.environ.get("GENOME_GZ_URL", DEFAULT_ZENODO_GENOME_GZ_URL).strip()

    if not args.checkpoint.is_file():
        _stream_download(ckpt_url, args.checkpoint, label="checkpoint")

    # If using a remote sequence API, we don't need any local genome file.
    if _use_sequence_api():
        return

    gz_dest = _genome_gz_destination(args.genome)
    if not gz_dest.is_file():
        _stream_download(gz_url, gz_dest, label="genome (.fa.gz)")


def fetch_window_from_sequence_api(
    chrom_key: str, window_start0: int, window_end_excl: int
) -> str:
    """
    Fetch reference DNA sequence for a window using a REST API.
    Default implementation uses Ensembl REST:
    GET /sequence/region/human/:region?content-type=text/plain
    region: {chrom}:{start}..{end}:1 (1-based inclusive, strand=1)
    """
    if window_end_excl <= window_start0:
        raise ValueError("基因组窗口无效：window_end_excl 必须大于 window_start0")
    base = os.environ.get("SEQ_API_BASE", DEFAULT_SEQ_API_BASE).rstrip("/")
    # Ensembl region is 1-based inclusive
    start_1b = window_start0 + 1
    end_1b = window_end_excl
    chrom = normalize_chrom(chrom_key)
    region = f"{chrom}:{start_1b}..{end_1b}:1"
    url = f"{base}/sequence/region/human/{region}?content-type=text/plain"
    req = Request(
        url,
        headers={
            "User-Agent": "MiRNA-web-predictor/1.0",
            "Accept": "text/plain",
        },
    )
    try:
        with urlopen(req, timeout=float(os.environ.get("SEQ_API_TIMEOUT_SEC", "30"))) as resp:
            text = resp.read().decode("utf-8", errors="replace").strip()
    except (URLError, OSError) as e:
        raise ValueError(f"序列 API 请求失败: {e}") from e
    if not text:
        raise ValueError("序列 API 返回空序列")
    return "".join(text.split()).upper()


def _add_cors_headers(handler: BaseHTTPRequestHandler) -> None:
    # Default to allowing any origin for easy frontend deployment.
    # You can lock this down on Render by setting CORS_ALLOW_ORIGIN to your Vercel domain.
    allow_origin = os.environ.get("CORS_ALLOW_ORIGIN", "*")
    handler.send_header("Access-Control-Allow-Origin", allow_origin)
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    _add_cors_headers(handler)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def _content_type_for_path(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    if ext == ".svg":
        return "image/svg+xml; charset=utf-8"
    if ext == ".css":
        return "text/css; charset=utf-8"
    if ext == ".js":
        return "application/javascript; charset=utf-8"
    return "application/octet-stream"


def _try_serve_web_static(handler: BaseHTTPRequestHandler, request_path: str) -> bool:
    """
    Serve static files that live alongside index.html (IMG_*.jpg, etc.)
    without requiring any frontend changes.
    """
    if not request_path.startswith("/"):
        return False
    rel = request_path.lstrip("/")
    # only allow single-segment paths like /IMG_7042.jpg
    if not rel or "/" in rel or "\\" in rel:
        return False
    candidate = (WEB_DIR / rel).resolve()
    try:
        candidate.relative_to(WEB_DIR)
    except ValueError:
        return False
    if not candidate.is_file():
        return False

    data = candidate.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", _content_type_for_path(candidate))
    handler.send_header("Cache-Control", "public, max-age=86400")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
    return True


def normalize_chrom(chrom: str) -> str:
    c = str(chrom).strip()
    if not c:
        raise ValueError("chrom 不能为空")
    if c.lower().startswith("chr"):
        c = c[3:]
    c = c.upper()
    if c == "M":
        c = "MT"
    return c


def normalize_strand(strand: str) -> str:
    s = str(strand).strip()
    if s not in {"+", "-"}:
        raise ValueError("strand 只能是 + 或 -")
    return s


def revcomp_dna(seq: str) -> str:
    return seq.translate(DNA_COMP)[::-1]


def build_seq_id(chrom: str, start_1b: int, end_1b: int, strand: str) -> str:
    inv = "_InvCom" if strand == "-" else ""
    return f"{chrom}_{start_1b}-{end_1b}{inv}_stem-{start_1b}-{end_1b}"


def ensure_model_state(args: argparse.Namespace) -> dict[str, object]:
    """只加载模型与配置；基因组按请求坐标只读局部窗口。"""
    with STATE_LOCK:
        if APP_STATE["ready"]:
            return APP_STATE

        dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, model_config = load_model(str(args.checkpoint), dev)
        use_mfe = bool(model_config.get("use_mfe", False))
        max_seq_len = int(model_config["max_seq_len"])
        padding_size = int(args.padding_size)
        premirna_len = max_seq_len - 2 * padding_size
        if premirna_len < 1:
            raise RuntimeError(
                f"max_seq_len={max_seq_len} 与 padding_size={padding_size} 不兼容"
            )
        if _use_sequence_api():
            gsrc = Path(f"SEQ_API:{os.environ.get('SEQ_API_BASE', DEFAULT_SEQ_API_BASE)}")
        else:
            gsrc = resolve_genome_source_path(args.genome)

        APP_STATE.update(
            {
                "ready": True,
                "model": model,
                "device": dev,
                "use_mfe": use_mfe,
                "padding_size": padding_size,
                "premirna_len": premirna_len,
                "max_seq_len": max_seq_len,
                "checkpoint": str(args.checkpoint),
                "genome_path": str(gsrc),
            }
        )
        return APP_STATE


def predict_one_region(
    *,
    chrom: str,
    start_1b: int,
    end_1b: int,
    strand: str,
    args: argparse.Namespace,
) -> dict:
    state = ensure_model_state(args)
    chrom_key = normalize_chrom(chrom)
    if start_1b < 1 or end_1b < 1:
        raise ValueError("坐标必须 >= 1")
    if end_1b < start_1b:
        raise ValueError("终点不能小于起点")

    pad = int(state["padding_size"])
    window_start0 = max(0, start_1b - 1 - pad)
    window_end_excl = end_1b + pad

    if _use_sequence_api():
        window = fetch_window_from_sequence_api(chrom_key, window_start0, window_end_excl)
    else:
        src_path = resolve_genome_source_path(args.genome)
        window = extract_chrom_window_from_fasta(
            src_path,
            chrom_key,
            window_start0,
            window_end_excl,
            progress=True,
        )

    strand_norm = normalize_strand(strand)
    s0l = (start_1b - 1) - window_start0
    e0l_excl = end_1b - window_start0
    if s0l < 0 or e0l_excl > len(window) or s0l >= e0l_excl:
        raise ValueError(
            "stem 区域超出已读取的基因组窗口（常见于染色体末端或窗口截断）"
        )
    dna = window[s0l:e0l_excl].upper()
    if not dna:
        raise ValueError("取到的 stem 序列为空")
    if strand_norm == "-":
        dna = revcomp_dna(dna)

    seq_id = build_seq_id(chrom_key, start_1b, end_1b, strand_norm)
    seq_data = {"original": dna}
    seq_data, pad_err = attach_padding_with_window(
        seq_id,
        seq_data,
        window,
        window_start0,
        pad,
        require_stem=True,
    )
    if pad_err:
        raise ValueError(f"padding 失败: {pad_err}")

    # Hairpin visualization inputs (sequence + dot-bracket structure).
    # We compute these once here for display; prediction encoding stays unchanged.
    try:
        import RNA  # ViennaRNA python bindings

        original_rna = dna.replace("T", "U").replace("t", "u")
        secondary_structure, mfe_vis = RNA.fold(original_rna)
    except Exception:
        secondary_structure, mfe_vis = "", float("nan")

    enc = encode_like_preprocess_positive(
        seq_data,
        padding_size=int(state["padding_size"]),
        premirna_len=int(state["premirna_len"]),
        struct_absent=float(STRUCT_CHANNEL_ABSENT),
    )
    if enc is None:
        raise ValueError("编码失败（可能含 N 或 RNA.fold 失败）")

    encoded, mfe = enc
    xt = torch.from_numpy(encoded[None, ...].astype(np.float32, copy=False))
    mfe_np = np.asarray([float(mfe)], dtype=np.float32)
    seq_len_np = np.full(1, float(state["max_seq_len"]), dtype=np.float32)
    probs_t = forward_batch(
        state["model"],
        xt,
        bool(state["use_mfe"]),
        mfe_np,
        seq_len_np,
        state["device"],
    )
    prob = float(probs_t.detach().cpu().numpy().reshape(-1)[0])

    hairpin_payload = get_hairpin_svg_or_enqueue(
        seq=original_rna,
        structure=secondary_structure,
        seq_id=seq_id,
    )

    return {
        "ok": True,
        "score": prob,
        "seq_id": seq_id,
        "chrom": chrom_key,
        "start_1b": int(start_1b),
        "end_1b": int(end_1b),
        "strand": strand_norm,
        "original_length": len(seq_data.get("original", "")),
        "upstream_length": len(seq_data.get("upstream", "")),
        "downstream_length": len(seq_data.get("downstream", "")),
        "mfe": float(mfe),
        "secondary_structure": secondary_structure,
        "original_sequence": original_rna,
        "hairpin_svg": str(hairpin_payload.get("hairpin_svg") or ""),
        "hairpin_svg_error": hairpin_payload.get("hairpin_svg_error"),
        "hairpin_key": hairpin_payload.get("hairpin_key"),
        "hairpin_pending": bool(hairpin_payload.get("hairpin_pending")),
        "use_mfe": bool(state["use_mfe"]),
        "padding_size": int(state["padding_size"]),
        "premirna_len": int(state["premirna_len"]),
        "max_seq_len": int(state["max_seq_len"]),
        "checkpoint": state["checkpoint"],
        "genome": state["genome_path"],
    }


class AppHandler(BaseHTTPRequestHandler):
    server_version = "MiRNAWebPredictor/1.0"

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        _add_cors_headers(self)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if _try_serve_web_static(self, parsed.path):
            return
        if parsed.path == "/":
            self.serve_index()
            return
        if parsed.path == "/api/hairpin":
            # GET /api/hairpin?key=...  -> {ok, pending, svg, error}
            try:
                from urllib.parse import parse_qs

                qs = parse_qs(parsed.query or "")
                key = (qs.get("key") or [""])[0].strip()
                if not re.fullmatch(r"[0-9a-f]{64}", key or ""):
                    json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid key"})
                    return

                cache_dir = Path(os.environ.get("R2DT_CACHE_DIR") or (WEB_DIR / ".r2dt_cache")).expanduser()
                cache_path = cache_dir / f"{key}.svg"
                if cache_path.is_file() and cache_path.stat().st_size > 0:
                    svg = cache_path.read_text(encoding="utf-8", errors="replace")
                    json_response(self, HTTPStatus.OK, {"ok": True, "pending": False, "svg": svg, "error": None})
                    return

                with _R2DT_JOB_LOCK:
                    info = dict(_R2DT_INFLIGHT.get(key) or {})
                pending = info.get("status") != "done"
                json_response(
                    self,
                    HTTPStatus.OK,
                    {"ok": True, "pending": bool(pending), "svg": "", "error": info.get("error")},
                )
                return
            except Exception as e:
                json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
                return
        if parsed.path == "/api/health":
            gsrc: Path | str = self.server.args.genome
            if _ASSETS_BOOTSTRAP_DONE.is_set() and _ASSETS_BOOTSTRAP_ERR is None:
                try:
                    gsrc = resolve_genome_source_path(self.server.args.genome)
                except FileNotFoundError:
                    gsrc = self.server.args.genome
            payload = {
                "ok": True,
                "ready": bool(APP_STATE["ready"]),
                "assets_ready": _ASSETS_BOOTSTRAP_DONE.is_set()
                and _ASSETS_BOOTSTRAP_ERR is None,
                "assets_error": _ASSETS_BOOTSTRAP_ERR,
                "sequence_mode": "api" if _use_sequence_api() else "local_genome",
                "checkpoint": str(self.server.args.checkpoint),
                "genome": str(self.server.args.genome),
                "genome_source": str(gsrc),
            }
            json_response(self, HTTPStatus.OK, payload)
            return
        json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
            return

        pending = _assets_ready_for_predict()
        if pending is not None:
            json_response(
                self,
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "error": pending},
            )
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8"))
            chrom = data.get("chrom", "")
            start_1b = int(data.get("start"))
            end_1b = int(data.get("end"))
            strand = data.get("strand", "+")
            result = predict_one_region(
                chrom=chrom,
                start_1b=start_1b,
                end_1b=end_1b,
                strand=strand,
                args=self.server.args,
            )
            json_response(self, HTTPStatus.OK, result)
        except Exception as e:
            json_response(
                self,
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": str(e)},
            )

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), fmt % args)
        )

    def serve_index(self) -> None:
        data = INDEX_HTML.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        _add_cors_headers(self)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class AppServer(ThreadingHTTPServer):
    def __init__(self, server_address, RequestHandlerClass, args: argparse.Namespace):
        super().__init__(server_address, RequestHandlerClass)
        self.args = args


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="坐标输入 -> 按现有 miRNA 模型返回 score 的本地网页服务")
    ap.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ["CHECKPOINT"]) if "CHECKPOINT" in os.environ else DEFAULT_CHECKPOINT,
    )
    ap.add_argument(
        "--genome",
        type=Path,
        default=Path(os.environ["GENOME"]) if "GENOME" in os.environ else DEFAULT_GENOME,
    )
    ap.add_argument("--padding_size", type=int, default=int(os.environ.get("PADDING_SIZE", "1000")))
    ap.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cpu"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    # 先绑定端口，再在后台拉取 Zenodo / 校验路径，避免 Render 端口探测超时。
    server = AppServer((args.host, args.port), AppHandler, args)
    threading.Thread(
        target=_bootstrap_assets_thread_body,
        args=(args,),
        daemon=True,
        name="assets-bootstrap",
    ).start()
    print(
        f"[web] listening http://{args.host}:{args.port} "
        f"(assets bootstrap in background)\n"
        f"[web] checkpoint={args.checkpoint}\n"
        f"[web] genome_arg={args.genome}",
        file=sys.stderr,
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] stopped", file=sys.stderr, flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
