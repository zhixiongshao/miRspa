#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
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

# Zenodo record: https://zenodo.org/records/19586578
# DOI: https://doi.org/10.5281/zenodo.19586578
DEFAULT_ZENODO_CHECKPOINT_URL = (
    "https://zenodo.org/records/19586578/files/best_model.pth?download=1"
)
DEFAULT_ZENODO_GENOME_GZ_URL = (
    "https://zenodo.org/records/19586578/files/"
    "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz?download=1"
)

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

    gz_dest = _genome_gz_destination(args.genome)
    if not gz_dest.is_file():
        _stream_download(gz_url, gz_dest, label="genome (.fa.gz)")


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
        if parsed.path == "/":
            self.serve_index()
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
