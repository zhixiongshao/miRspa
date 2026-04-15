#!/usr/bin/env python3
"""
对编码序列做逐位遮挡（occlusion），计算正类概率变化：delta = prob_original - prob_masked。
每条序列单独输出 4 个 CSV：
  - *_position_saliency.csv：score 对输入 x 求导，每位置 sum_c |∂score/∂x_{c,p}|
  - *_occlusion_delta_mask_base_only.csv
  - *_occlusion_delta_mask_struct_only.csv
  - *_occlusion_delta_mask_base_and_struct.csv
文件名前缀为「序号_净化后的序列名」，避免多条序列互相覆盖。
所有遮挡前向一律 batch_size=1。
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_seq_stem(name: str, max_len: int = 120) -> str:
    """序列名中的路径/非法字符转为下划线，用作文件名的一部分。"""
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("._")
    return (s or "seq")[:max_len]


def resolve_indices_by_mir_tokens(names: list[str], tokens: list[str]) -> list[int]:
    """
    每个 token（如 mir-21）在 names 中找一条序列：名称需包含该 miRNA 编号，
    且避免 mir-21 误匹配 mir-210（要求 token 后为非数字或结尾）。
    多条命中时取第一条并打日志。
    """
    out: list[int] = []
    for tok in tokens:
        esc = re.escape(tok)
        pat = re.compile(rf"(?i){esc}(?:$|[^0-9])")
        hits = [i for i, n in enumerate(names) if pat.search(n)]
        if not hits:
            raise ValueError(
                f"未找到名称匹配「{tok}」的序列。JSON 内键名示例（前 5 个）: {names[:5]}"
            )
        if len(hits) > 1:
            logger.warning(
                "「%s」命中 %d 条，使用第一条: %s",
                tok,
                len(hits),
                names[hits[0]],
            )
        out.append(hits[0])
    return out


def load_model(checkpoint_path: str, device: torch.device):
    """与 test_mlp_classifier 一致的加载逻辑。"""
    # Import inference-only definition to avoid pulling in heavy test deps
    # (sklearn/matplotlib) during web deployment.
    from mlp_classifier_model import MLPClassifier

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    model_type = model_config.get("model_type", "mlp")
    use_mfe = model_config.get("use_mfe", False)
    use_transformer = model_config.get("use_transformer", False)
    nhead = model_config.get("nhead") or 8
    num_transformer_blocks = model_config.get("num_transformer_blocks") or 3

    if model_type == "resnet18":
        from train_mlp_classifier import ResNet18_1D

        model = ResNet18_1D(
            input_channels=model_config["input_dim"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
            use_mfe=use_mfe,
            use_transformer=use_transformer,
            nhead=nhead,
            num_transformer_blocks=num_transformer_blocks,
        )
    elif model_type == "resnet50":
        from train_mlp_classifier import ResNet50_1D

        model = ResNet50_1D(
            input_channels=model_config["input_dim"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
            use_mfe=use_mfe,
        )
    elif model_type == "resnet101":
        from train_mlp_classifier import ResNet101_1D

        model = ResNet101_1D(
            input_channels=model_config["input_dim"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
            use_mfe=use_mfe,
        )
    elif model_type == "mlp":
        model = MLPClassifier(
            input_dim=model_config["input_dim"],
            max_seq_len=model_config["max_seq_len"],
            hidden_dims=model_config["hidden_dims"],
            num_classes=model_config["num_classes"],
            dropout=model_config["dropout"],
        )
    elif model_type == "transformer":
        from train_mlp_classifier import PureTransformer1D

        model = PureTransformer1D(
            input_channels=model_config["input_dim"],
            num_classes=model_config["num_classes"],
            d_model=model_config.get("d_model", 128),
            num_layers=num_transformer_blocks,
            nhead=nhead,
            dropout=model_config["dropout"],
            use_mfe=use_mfe,
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, model_config


def positive_class_prob(out: torch.Tensor, model: torch.Module) -> torch.Tensor:
    """与 test 一致：二分类正类概率。"""
    if out.dim() == 2 and out.shape[1] == 2:
        return torch.softmax(out, dim=1)[:, 1]
    if len(out.shape) > 1 and out.shape[1] == 1:
        out = out.squeeze(1)
    use_sig = getattr(model, "use_sigmoid", False)
    if use_sig:
        return out
    return torch.sigmoid(out)


def gradient_saliency_per_position(
    model: torch.Module,
    x: torch.Tensor,
    use_mfe: bool,
    mfe_np: np.ndarray,
    seq_len_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    score = 正类概率 P(y=1|x)，对 x 反向传播。
    每个位置 p：saliency(p) = sum_{c=0}^{4} |∂score/∂x_{c,p}|，形状 (L,)。
    """
    model.zero_grad(set_to_none=True)
    xc = x.clone().detach().to(device).requires_grad_(True)
    if use_mfe:
        out = model(xc, mfe=mfe_np, seq_len=seq_len_np)
    else:
        out = model(xc)
    score = positive_class_prob(out, model)
    if score.dim() > 0:
        score = score.sum()
    score.backward()
    if xc.grad is None:
        raise RuntimeError("输入梯度为空，请检查模型与前向是否在计算图中。")
    # (1, 5, L) -> 对通道维 |.| 求和 -> (L,)
    sal = xc.grad.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
    return sal


@torch.no_grad()
def forward_batch(
    model: torch.Module,
    x: torch.Tensor,
    use_mfe: bool,
    mfe_np: np.ndarray | None,
    seq_len_np: np.ndarray | None,
    device: torch.device,
) -> torch.Tensor:
    x = x.to(device).contiguous()
    if use_mfe:
        out = model(x, mfe=mfe_np, seq_len=seq_len_np)
    else:
        out = model(x)
    return positive_class_prob(out, model)


def apply_mask(
    base: torch.Tensor,
    positions: np.ndarray,
    mode: str,
    struct_absent: float,
) -> torch.Tensor:
    """
    base: (1, 5, L)
    positions: (B,) 要遮挡的列索引
    mode: base | struct | both
    返回 (B, 5, L)
    """
    bsz = len(positions)
    L = base.shape[2]
    batch = base.expand(bsz, -1, -1).clone()
    for k, pos in enumerate(positions):
        p = int(pos)
        if p < 0 or p >= L:
            continue
        if mode == "base":
            batch[k, :4, p] = 0.0
        elif mode == "struct":
            batch[k, 4, p] = struct_absent
        elif mode == "both":
            batch[k, :4, p] = 0.0
            batch[k, 4, p] = struct_absent
        else:
            raise ValueError(mode)
    return batch


def run_occlusion(
    json_path: Path,
    ckpt_path: Path,
    out_dir: Path,
    device: str,
    struct_absent: float,
    *,
    all_sequences: bool,
    only_mir_tokens: list[str] | None,
    max_seqs: int | None,
):
    from test_mlp_classifier import MiRNAClassificationDataset

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(str(ckpt_path), dev)
    use_mfe = cfg.get("use_mfe", False)
    max_seq_len = cfg["max_seq_len"]

    ds = MiRNAClassificationDataset(str(json_path), max_seq_len=max_seq_len, pos_neg_ratio=None)
    if all_sequences:
        indices = list(range(len(ds)))
        if max_seqs is not None:
            indices = indices[: max_seqs]
    else:
        if not only_mir_tokens:
            raise ValueError("未指定 only_mir_tokens 且未使用 --all_sequences")
        indices = resolve_indices_by_mir_tokens(ds.names, only_mir_tokens)
    L = max_seq_len
    logger.info("将处理 %d 条序列: %s", len(indices), [ds.names[i] for i in indices])

    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["seq_id", "position", "prob_original", "prob_masked", "delta"]
    occlusion_suffix = {
        "base": "occlusion_delta_mask_base_only",
        "struct": "occlusion_delta_mask_struct_only",
        "both": "occlusion_delta_mask_base_and_struct",
    }
    written: list[Path] = []

    for run_i, idx in enumerate(tqdm(indices, desc="sequences")):
        seq_t, label, _, mfe, seq_len = ds[idx]
        name = ds.names[idx]
        prefix = f"{run_i:04d}_{safe_seq_stem(name)}"
        x = seq_t.unsqueeze(0)  # (1,5,L)

        mfe_np = np.array([mfe], dtype=np.float32)
        seq_len_np = np.array([seq_len], dtype=np.float32)

        p0 = forward_batch(model, x, use_mfe, mfe_np, seq_len_np, dev)
        prob_orig = float(p0.item())

        with torch.set_grad_enabled(True):
            grad_sal = gradient_saliency_per_position(
                model, x, use_mfe, mfe_np, seq_len_np, dev
            )

        sal_path = out_dir / f"{prefix}_position_saliency.csv"
        with open(sal_path, "w", newline="", encoding="utf-8") as sf:
            sw = csv.DictWriter(
                sf,
                fieldnames=["seq_id", "position", "sum_abs_grad_channels"],
            )
            sw.writeheader()
            for p in range(L):
                sw.writerow(
                    {
                        "seq_id": name,
                        "position": int(p),
                        "sum_abs_grad_channels": f"{grad_sal[p]:.10e}",
                    }
                )
        written.append(sal_path)

        for mode_key in ("base", "struct", "both"):
            oc_path = out_dir / f"{prefix}_{occlusion_suffix[mode_key]}.csv"
            with open(oc_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for p in range(L):
                    pos = np.array([p], dtype=np.int64)
                    batch_x = apply_mask(x, pos, mode_key, struct_absent)
                    probs_m = forward_batch(model, batch_x, use_mfe, mfe_np, seq_len_np, dev)
                    pm = float(probs_m.item())
                    delta = prob_orig - pm
                    w.writerow(
                        {
                            "seq_id": name,
                            "position": int(p),
                            "prob_original": f"{prob_orig:.8f}",
                            "prob_masked": f"{pm:.8f}",
                            "delta": f"{delta:.8f}",
                        }
                    )
            written.append(oc_path)

    logger.info(
        "每条序列 4 个表：position_saliency（sum_c|∂score/∂x|）与三种 occlusion delta；"
        "共 %d 条序列，%d 个文件，目录: %s",
        len(indices),
        len(written),
        out_dir,
    )
    for p in written[:20]:
        logger.info("  %s", p.name)
    if len(written) > 20:
        logger.info("  ... 另有 %d 个文件", len(written) - 20)


def main():
    parser = argparse.ArgumentParser(description="Occlusion saliency for MLP/ResNet miRNA classifier")
    parser.add_argument(
        "--json",
        type=str,
        default="/mnt/data/home/zhixiong/MiRNA/github/padding_1000bp_dataset.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/data/home/zhixiong/MiRNA/github/dataset/miRspa1000/best_model.pth",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/mnt/data/home/zhixiong/MiRNA/github/dataset/miRspa1000/occlusion_saliency",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--max_seqs",
        type=int,
        default=None,
        help="与 --all_sequences 联用：只处理排序后的前 N 条",
    )
    parser.add_argument(
        "--only_mir_tokens",
        nargs="+",
        default=["mir-21", "mir-155"],
        help="默认只处理名称匹配这些 miRNA 的序列（各取一条）；与 --all_sequences 互斥",
    )
    parser.add_argument(
        "--all_sequences",
        action="store_true",
        help="处理 JSON 中全部序列（忽略 --only_mir_tokens）",
    )
    parser.add_argument(
        "--struct_absent",
        type=float,
        default=2.0,
        help="结构通道遮挡取值（与预处理一致，默认 2）",
    )
    args = parser.parse_args()
    if args.all_sequences:
        logger.info("模式: 处理 JSON 中全部序列（可用 --max_seqs 截断）")
    else:
        logger.info("模式: 仅处理匹配 --only_mir_tokens 的序列: %s", args.only_mir_tokens)
    run_occlusion(
        Path(args.json),
        Path(args.checkpoint),
        Path(args.out_dir),
        args.device,
        args.struct_absent,
        all_sequences=args.all_sequences,
        only_mir_tokens=args.only_mir_tokens,
        max_seqs=args.max_seqs,
    )


if __name__ == "__main__":
    main()
