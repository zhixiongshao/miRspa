#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference-only model components used for loading checkpoints in production.

This module avoids importing training scripts that pull in optional heavy deps
like sklearn/matplotlib which are unnecessary for serving.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    """1D ResNet basic block (for ResNet18)."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck1D(nn.Module):
    """1D ResNet bottleneck block (for ResNet50/101)."""

    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class TransformerBlock1D(nn.Module):
    """1D Transformer block used in this project."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class PureTransformer1D(nn.Module):
    """Transformer-only 1D classifier (no ResNet backbone)."""

    def __init__(
        self,
        input_channels: int = 5,
        num_classes: int = 2,
        d_model: int = 128,
        num_layers: int = 3,
        nhead: int = 8,
        dropout: float = 0.3,
        use_mfe: bool = False,
    ) -> None:
        super().__init__()
        self.use_mfe = use_mfe
        self.input_proj = nn.Conv1d(input_channels, d_model, kernel_size=1, bias=False)
        self.input_bn = nn.BatchNorm1d(d_model)
        self.input_relu = nn.ReLU(inplace=True)
        dim_feedforward = d_model * 4
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock1D(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        fc_input_dim = d_model + (1 if use_mfe else 0)
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x: torch.Tensor, mfe=None, seq_len=None) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_relu(x)
        x = x.permute(0, 2, 1)  # (B, L, d_model)
        for block in self.transformer_blocks:
            x = block(x)
        x = x.mean(dim=1)
        if self.use_mfe:
            if mfe is None or seq_len is None:
                raise ValueError("use_mfe=True时，必须提供mfe和seq_len参数")
            if not torch.is_tensor(mfe):
                mfe = torch.tensor(mfe, dtype=x.dtype, device=x.device)
            else:
                mfe = mfe.to(dtype=x.dtype, device=x.device)
            if not torch.is_tensor(seq_len):
                seq_len = torch.tensor(seq_len, dtype=x.dtype, device=x.device)
            else:
                seq_len = seq_len.to(dtype=x.dtype, device=x.device)
            dg = -mfe / (seq_len + 1e-8)
            if dg.dim() == 1:
                dg = dg.unsqueeze(1)
            x = torch.cat([x, dg], dim=1)
        x = self.dropout(x)
        return self.fc(x)


class ResNet1D(nn.Module):
    """1D ResNet classifier supporting ResNet18/50/101 variants."""

    def __init__(
        self,
        input_channels: int = 5,
        num_classes: int = 2,
        dropout: float = 0.3,
        layers: list[int] | None = None,
        block=BasicBlock1D,
        use_mfe: bool = False,
        use_transformer: bool = False,
        nhead: int = 8,
        num_transformer_blocks: int = 3,
    ) -> None:
        super().__init__()
        if layers is None:
            layers = [2, 2, 2, 2]
        self.block = block
        self.in_channels = 64
        self.use_mfe = use_mfe
        self.use_transformer = use_transformer

        self.conv1 = nn.Conv1d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        if use_transformer:
            d_model = 512 * block.expansion
            dim_feedforward = d_model * 4
            self.transformer_blocks = nn.ModuleList(
                [
                    TransformerBlock1D(d_model, nhead, dim_feedforward, dropout)
                    for _ in range(num_transformer_blocks)
                ]
            )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        fc_input_dim = 512 * block.expansion
        if use_mfe:
            fc_input_dim += 1
            self.mfe_relu = nn.ReLU(inplace=True)
            self.mfe_bn = nn.BatchNorm1d(fc_input_dim)

        self.use_sigmoid = use_mfe
        if self.use_sigmoid:
            self.fc = nn.Linear(fc_input_dim, 1)
        else:
            self.fc = nn.Linear(fc_input_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels * self.block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels * self.block.expansion),
            )
        layers: list[nn.Module] = []
        layers.append(self.block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * self.block.expansion
        for _ in range(1, blocks):
            layers.append(self.block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, mfe=None, seq_len=None) -> torch.Tensor:
        x = self.conv1(x).contiguous()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.use_transformer:
            x = x.permute(0, 2, 1)
            for block in self.transformer_blocks:
                x = block(x)
            x = x.permute(0, 2, 1)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.use_mfe:
            if mfe is None or seq_len is None:
                raise ValueError("use_mfe=True时，必须提供mfe和seq_len参数")
            mfe_tensor = torch.tensor(mfe, dtype=x.dtype, device=x.device)
            seq_len_tensor = torch.tensor(seq_len, dtype=x.dtype, device=x.device)
            dg = -mfe_tensor / (seq_len_tensor + 1e-8)
            dg = dg.unsqueeze(1) if dg.dim() == 1 else dg
            x = torch.cat([x, dg], dim=1)
            x = self.mfe_relu(x)
            x = self.mfe_bn(x)

        x = self.dropout(x)
        x = self.fc(x)
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        return x


def ResNet18_1D(
    input_channels: int = 5,
    num_classes: int = 2,
    dropout: float = 0.3,
    use_mfe: bool = False,
    use_transformer: bool = False,
    nhead: int = 8,
    num_transformer_blocks: int = 3,
) -> ResNet1D:
    return ResNet1D(
        input_channels,
        num_classes,
        dropout,
        layers=[2, 2, 2, 2],
        block=BasicBlock1D,
        use_mfe=use_mfe,
        use_transformer=use_transformer,
        nhead=nhead,
        num_transformer_blocks=num_transformer_blocks,
    )


def ResNet50_1D(
    input_channels: int = 5,
    num_classes: int = 2,
    dropout: float = 0.3,
    use_mfe: bool = False,
) -> ResNet1D:
    return ResNet1D(
        input_channels,
        num_classes,
        dropout,
        layers=[3, 4, 6, 3],
        block=Bottleneck1D,
        use_mfe=use_mfe,
    )


def ResNet101_1D(
    input_channels: int = 5,
    num_classes: int = 2,
    dropout: float = 0.3,
    use_mfe: bool = False,
) -> ResNet1D:
    return ResNet1D(
        input_channels,
        num_classes,
        dropout,
        layers=[3, 4, 23, 3],
        block=Bottleneck1D,
        use_mfe=use_mfe,
    )

