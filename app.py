#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from occlusion_saliency_mlp import forward_batch, load_model
from preprocess_padding_dataset import STRUCT_CHANNEL_ABSENT
from stream_predict_padding_fasta import (
    attach_padding_from_genome,
    encode_like_preprocess_positive,
)
from verify_stem_padding_vs_genome import load_genome


DEFAULT_CHECKPOINT = ROOT / "github/dataset/miRspa1000/best_model.pth"
DEFAULT_GENOME = ROOT / "gene/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
INDEX_HTML = Path(__file__).with_name("index.html")

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
    "genome": None,
    "checkpoint": None,
    "genome_path": None,
}

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


def fetch_genome_dna(genome: dict[str, str], chrom: str, start_1b: int, end_1b: int) -> str:
    ch = genome.get(chrom)
    if ch is None:
        raise ValueError(f"基因组里找不到染色体: {chrom}")
    if start_1b < 1 or end_1b < 1:
        raise ValueError("坐标必须 >= 1")
    if end_1b < start_1b:
        raise ValueError("终点不能小于起点")
    s0 = start_1b - 1
    e0 = end_1b
    if e0 > len(ch):
        raise ValueError(f"终点超出染色体长度 {len(ch):,}")
    dna = ch[s0:e0].upper()
    if not dna:
        raise ValueError("取到的序列为空")
    return dna


def get_state(args: argparse.Namespace) -> dict[str, object]:
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
        genome = load_genome(args.genome, progress=True)

        APP_STATE.update(
            {
                "ready": True,
                "model": model,
                "device": dev,
                "use_mfe": use_mfe,
                "padding_size": padding_size,
                "premirna_len": premirna_len,
                "max_seq_len": max_seq_len,
                "genome": genome,
                "checkpoint": str(args.checkpoint),
                "genome_path": str(args.genome),
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
    state = get_state(args)
    genome = state["genome"]
    assert isinstance(genome, dict)

    chrom_key = normalize_chrom(chrom)
    strand_norm = normalize_strand(strand)
    dna = fetch_genome_dna(genome, chrom_key, start_1b, end_1b)
    if strand_norm == "-":
        dna = revcomp_dna(dna)

    seq_id = build_seq_id(chrom_key, start_1b, end_1b, strand_norm)
    seq_data = {"original": dna}
    seq_data, pad_err = attach_padding_from_genome(
        seq_id,
        seq_data,
        genome,
        int(state["padding_size"]),
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
            payload = {
                "ok": True,
                "ready": bool(APP_STATE["ready"]),
                "checkpoint": str(self.server.args.checkpoint),
                "genome": str(self.server.args.genome),
            }
            json_response(self, HTTPStatus.OK, payload)
            return
        json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found"})
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
    if not args.checkpoint.is_file():
        raise SystemExit(f"找不到 checkpoint: {args.checkpoint}")
    if not args.genome.is_file():
        raise SystemExit(f"找不到 genome: {args.genome}")

    server = AppServer((args.host, args.port), AppHandler, args)
    print(
        f"[web] serving http://{args.host}:{args.port}\n"
        f"[web] checkpoint={args.checkpoint}\n"
        f"[web] genome={args.genome}",
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
