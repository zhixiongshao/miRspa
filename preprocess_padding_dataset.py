#!/usr/bin/env python3
"""
处理带 padding 或仅 pre-miRNA 的数据集；--pos_fasta / --neg_fasta 可只给其一（至少一个）。

FASTA：
- 三段式：>id|original / |upstream_padding_NNbp / |downstream_padding_NNbp；
- 或单条式（如 0bp、无侧翼）：>header，序列整段作为 original，上下游为空。

处理流程：
1. 使用 RNAfold 预测 premiRNA（original）的二级结构与 MFE；
2. 先拼接 upstream + premiRNA + downstream，再整体左侧补到 total_len = 2*padding_size + premirna_len（补位为碱基 0 + 结构通道 struct_absent，默认 2）；
3. 侧翼仅有碱基 one-hot，结构通道为 STRUCT_CHANNEL_ABSENT；
4. 输出 JSON。

结构通道：'('=1, ')'=-1, '.'=0；2 表示无结构语义（侧翼、左侧补位）。
"""
import RNA
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import re

# 结构通道：2 表示无有效碱基或无结构语义（侧翼、pre-miRNA 左侧补到 premirna_len 的占位）；「.」仍为 0
STRUCT_CHANNEL_ABSENT = 2.0


def read_fasta_with_padding(fasta_file):
    """
    读取 FASTA（三段式带 padding 或 单条序列不加 padding 均可）。

    三段式：>seq_id|original / |upstream_padding_NNbp / |downstream_padding_NNbp
    单条式：>任意头（无上述 | 字段）——整段序列记为 original，upstream/downstream 为空字符串。
    """
    sequences = {}
    current_id = None
    current_type = None
    current_seq = []

    def flush():
        nonlocal current_id, current_type, current_seq
        if current_id is not None and current_type is not None:
            sequences[current_id][current_type] = "".join(current_seq).upper()
        current_seq = []

    with open(fasta_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                flush()
                rest = line[1:].strip()
                # 与 triplet 识别：必须含 |original 或 padding 后缀之一
                if (
                    "|original" in rest
                    or "|upstream_padding_" in rest
                    or "|downstream_padding_" in rest
                ):
                    if "|" not in rest:
                        current_id = None
                        current_type = None
                        continue
                    seq_id, seq_type = rest.split("|", 1)
                    if seq_id not in sequences:
                        sequences[seq_id] = {}
                    if "original" in seq_type:
                        current_type = "original"
                    elif "upstream_padding" in seq_type:
                        current_type = "upstream"
                    elif "downstream_padding" in seq_type:
                        current_type = "downstream"
                    else:
                        current_type = None
                        current_id = None
                        continue
                    current_id = seq_id
                else:
                    # 普通 FASTA：一条记录 = 一个 original，无侧翼
                    seq_id = rest
                    if seq_id not in sequences:
                        sequences[seq_id] = {}
                    current_id = seq_id
                    current_type = "original"
            elif line and current_id is not None and current_type is not None:
                current_seq.append(line)

        flush()

    return sequences

def dna_to_rna(seq):
    """将DNA序列转换为RNA序列"""
    return seq.replace('T', 'U').replace('t', 'u')

def encode_base(base):
    """对单个碱基进行独热编码"""
    base = base.upper()
    if base == 'A':
        return [1, 0, 0, 0]
    elif base == 'U':
        return [0, 1, 0, 0]
    elif base == 'G':
        return [0, 0, 1, 0]
    elif base == 'C':
        return [0, 0, 0, 1]
    else:
        return [0, 0, 0, 0]

def encode_sequence_only(sequence):
    """
    只编码序列（碱基），不包含结构信息
    返回: (4, seq_len)的numpy数组
    """
    seq_len = len(sequence)
    encoded = np.zeros((4, seq_len), dtype=np.float32)
    
    for i, base in enumerate(sequence):
        base_encoding = encode_base(base)
        encoded[0, i] = base_encoding[0]  # A
        encoded[1, i] = base_encoding[1]  # U
        encoded[2, i] = base_encoding[2]  # G
        encoded[3, i] = base_encoding[3]  # C
    
    return encoded

def encode_sequence_with_structure(sequence, structure):
    """
    编码序列和结构（用于premiRNA部分）
    返回: (5, seq_len)的numpy数组
    """
    seq_len = len(sequence)
    
    if len(structure) != seq_len:
        raise ValueError(f"Sequence length ({seq_len}) != structure length ({len(structure)})")
    
    encoded = np.zeros((5, seq_len), dtype=np.float32)
    
    for i, (base, struct) in enumerate(zip(sequence, structure)):
        base_encoding = encode_base(base)
        struct_encoding = 1 if struct == '(' else (-1 if struct == ')' else 0)
        
        encoded[0, i] = base_encoding[0]  # A
        encoded[1, i] = base_encoding[1]  # U
        encoded[2, i] = base_encoding[2]  # G
        encoded[3, i] = base_encoding[3]  # C
        encoded[4, i] = struct_encoding   # Structure
    
    return encoded

def build_full_encoded_sequence(
    upstream_rna: str,
    premirna_rna: str,
    premirna_structure: str,
    downstream_rna: str,
    target_total_len: int,
    struct_absent: float,
) -> np.ndarray:
    """
    先拼接 upstream + premiRNA + downstream，再整体左侧补位到 target_total_len。
    左侧补位值：碱基通道全 0、结构通道 struct_absent（默认 2）。
    """
    # upstream 编码：仅碱基，结构通道统一设为 struct_absent
    up_len = len(upstream_rna)
    up_encoded = np.zeros((5, up_len), dtype=np.float32)
    if up_len > 0:
        up_encoded[0:4, :] = encode_sequence_only(upstream_rna)
        up_encoded[4, :] = float(struct_absent)

    # premiRNA 编码：碱基 + RNAfold 二级结构
    premirna_encoded = encode_sequence_with_structure(premirna_rna, premirna_structure)

    # downstream 编码：仅碱基，结构通道统一设为 struct_absent
    dn_len = len(downstream_rna)
    dn_encoded = np.zeros((5, dn_len), dtype=np.float32)
    if dn_len > 0:
        dn_encoded[0:4, :] = encode_sequence_only(downstream_rna)
        dn_encoded[4, :] = float(struct_absent)

    combined = np.concatenate([up_encoded, premirna_encoded, dn_encoded], axis=1)
    current_len = combined.shape[1]

    # 超长时截断；不足时整体左补位
    if current_len >= target_total_len:
        return combined[:, :target_total_len]

    padded = np.zeros((5, target_total_len), dtype=np.float32)
    pad_len = target_total_len - current_len
    padded[4, :pad_len] = float(struct_absent)
    padded[:, pad_len:] = combined
    return padded

def process_padding_dataset(
    pos_fasta_file,
    neg_fasta_file,
    padding_size,
    premirna_len=180,
    output_file=None,
    struct_absent=STRUCT_CHANNEL_ABSENT,
):
    """
    处理带padding的数据集

    Args:
        pos_fasta_file: 正样本 FASTA；可为 None（仅处理负样本）
        neg_fasta_file: 负样本 FASTA；可为 None（仅处理正样本）
        padding_size: padding长度（用于验证）
        premirna_len: premiRNA序列的目标长度（默认180）
        output_file: 输出JSON文件路径
        struct_absent: 无碱基/侧翼区结构通道取值（默认 2.0）
    """
    print("="*70)
    print("处理带padding的数据集")
    print("="*70)
    print(f"Padding大小: {padding_size} bp")
    print(f"Pre-miRNA目标长度: {premirna_len} bp")
    print(f"结构通道「无/填充」取值: {struct_absent}")
    print(f"总序列长度: {padding_size} + {premirna_len} + {padding_size} = {padding_size * 2 + premirna_len} bp")
    if pos_fasta_file is None:
        print("正样本: 未指定 --pos_fasta，跳过")
    if neg_fasta_file is None:
        print("负样本: 未指定 --neg_fasta，跳过")

    all_data = {}
    stats = {
        'pos_total': 0,
        'pos_processed': 0,
        'pos_failed': 0,
        'neg_total': 0,
        'neg_processed': 0,
        'neg_failed': 0
    }
    
    # 处理正样本
    if pos_fasta_file is not None:
        print(f"\n处理正样本文件: {pos_fasta_file}")
        pos_sequences = read_fasta_with_padding(pos_fasta_file)
        stats['pos_total'] = len(pos_sequences)
        print(f"读取到 {stats['pos_total']:,} 条正样本序列")

        print(f"使用RNAfold预测premiRNA序列的二级结构和MFE...")
        for seq_id, seq_data in tqdm(pos_sequences.items(), desc="正样本处理"):
            if 'original' not in seq_data:
                stats['pos_failed'] += 1
                continue

            original_seq = seq_data['original']
            upstream_seq = seq_data.get('upstream', '')
            downstream_seq = seq_data.get('downstream', '')

            # 转换为RNA
            original_rna = dna_to_rna(original_seq)

            # 跳过包含N的序列
            if 'N' in original_rna.upper():
                stats['pos_failed'] += 1
                continue

            try:
                # 对premiRNA序列进行RNAfold预测
                structure, mfe = RNA.fold(original_rna)

                if len(structure) != len(original_rna):
                    stats['pos_failed'] += 1
                    continue

                upstream_rna = dna_to_rna(upstream_seq) if upstream_seq else ''
                downstream_rna = dna_to_rna(downstream_seq) if downstream_seq else ''

                total_len = padding_size + premirna_len + padding_size
                full_encoded = build_full_encoded_sequence(
                    upstream_rna=upstream_rna,
                    premirna_rna=original_rna,
                    premirna_structure=structure,
                    downstream_rna=downstream_rna,
                    target_total_len=total_len,
                    struct_absent=struct_absent,
                )

                # 存储数据
                all_data[seq_id] = {
                    'sequence': original_rna,
                    'structure': structure,
                    'mfe': float(mfe),
                    'encoded_sequence': full_encoded.tolist(),
                    'label': 1,
                    'premirna_length': len(original_rna),
                    'upstream_length': len(upstream_rna),
                    'downstream_length': len(downstream_rna)
                }

                stats['pos_processed'] += 1

            except Exception as e:
                stats['pos_failed'] += 1
                continue

    # 处理负样本
    if neg_fasta_file is not None:
        print(f"\n处理负样本文件: {neg_fasta_file}")
        neg_sequences = read_fasta_with_padding(neg_fasta_file)
        stats['neg_total'] = len(neg_sequences)
        print(f"读取到 {stats['neg_total']:,} 条负样本序列")

        print(f"使用RNAfold预测premiRNA序列的二级结构和MFE...")
        for seq_id, seq_data in tqdm(neg_sequences.items(), desc="负样本处理"):
            if 'original' not in seq_data:
                stats['neg_failed'] += 1
                continue

            original_seq = seq_data['original']
            upstream_seq = seq_data.get('upstream', '')
            downstream_seq = seq_data.get('downstream', '')

            # 转换为RNA
            original_rna = dna_to_rna(original_seq)

            # 跳过包含N的序列
            if 'N' in original_rna.upper():
                stats['neg_failed'] += 1
                continue

            try:
                # 对premiRNA序列进行RNAfold预测
                structure, mfe = RNA.fold(original_rna)

                if len(structure) != len(original_rna):
                    stats['neg_failed'] += 1
                    continue

                upstream_rna = dna_to_rna(upstream_seq) if upstream_seq else ''
                downstream_rna = dna_to_rna(downstream_seq) if downstream_seq else ''

                total_len = padding_size + premirna_len + padding_size
                full_encoded = build_full_encoded_sequence(
                    upstream_rna=upstream_rna,
                    premirna_rna=original_rna,
                    premirna_structure=structure,
                    downstream_rna=downstream_rna,
                    target_total_len=total_len,
                    struct_absent=struct_absent,
                )

                # 存储数据
                all_data[seq_id] = {
                    'sequence': original_rna,
                    'structure': structure,
                    'mfe': float(mfe),
                    'encoded_sequence': full_encoded.tolist(),
                    'label': 0,
                    'premirna_length': len(original_rna),
                    'upstream_length': len(upstream_rna),
                    'downstream_length': len(downstream_rna)
                }

                stats['neg_processed'] += 1

            except Exception as e:
                stats['neg_failed'] += 1
                continue

    if not all_data:
        print("\n错误: 未处理任何序列（请至少提供 --pos_fasta 或 --neg_fasta 之一，且 FASTA 可读）")
        return

    # 保存JSON文件
    output_path = Path(output_file) if output_file else Path(f"dataset/padding_{padding_size}bp_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存JSON文件: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    # 打印统计信息
    print(f"\n" + "="*70)
    print("处理完成统计")
    print("="*70)
    print(f"正样本:")
    print(f"  总数: {stats['pos_total']:,}")
    print(f"  成功处理: {stats['pos_processed']:,}")
    print(f"  失败: {stats['pos_failed']:,}")
    print(f"\n负样本:")
    print(f"  总数: {stats['neg_total']:,}")
    print(f"  成功处理: {stats['neg_processed']:,}")
    print(f"  失败: {stats['neg_failed']:,}")
    print(f"\n总计:")
    print(f"  总序列数: {len(all_data):,}")
    print(f"  正样本: {sum(1 for v in all_data.values() if v['label'] == 1):,}")
    print(f"  负样本: {sum(1 for v in all_data.values() if v['label'] == 0):,}")
    print(f"\n输出文件: {output_path}")
    
    # 验证编码维度
    if all_data:
        sample_key = list(all_data.keys())[0]
        sample_data = all_data[sample_key]
        encoded_shape = np.array(sample_data['encoded_sequence']).shape
        print(f"\n编码维度验证:")
        print(f"  编码数组形状: {encoded_shape}")
        print(f"  预期形状: (5, {padding_size * 2 + premirna_len})")
        if encoded_shape == (5, padding_size * 2 + premirna_len):
            print(f"  ✅ 维度验证通过")
        else:
            print(f"  ⚠️  维度不匹配")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理带padding的数据集")
    parser.add_argument("--pos_fasta", type=str, default=None,
                       help="正样本FASTA（可省略，但至少需与 --neg_fasta 二选一）")
    parser.add_argument("--neg_fasta", type=str, default=None,
                       help="负样本FASTA（可省略，但至少需与 --pos_fasta 二选一）")
    parser.add_argument("--padding_size", type=int, required=True,
                       help="Padding大小（bp）")
    parser.add_argument("--premirna_len", type=int, default=180,
                       help="Pre-miRNA目标长度（默认180）")
    parser.add_argument("--struct_absent", type=float, default=STRUCT_CHANNEL_ABSENT,
                       help="无碱基区与基因组侧翼的结构通道取值（默认2，与「.」的0区分）")
    parser.add_argument("--output_file", type=str, default=None,
                       help="输出JSON文件路径（默认：dataset/padding_{padding_size}bp_dataset.json）")
    
    args = parser.parse_args()

    if args.pos_fasta is None and args.neg_fasta is None:
        parser.error("请至少提供 --pos_fasta 或 --neg_fasta 之一")

    process_padding_dataset(
        args.pos_fasta,
        args.neg_fasta,
        args.padding_size,
        args.premirna_len,
        args.output_file,
        struct_absent=args.struct_absent,
    )
