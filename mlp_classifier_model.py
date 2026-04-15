#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference-only model definitions.

This module exists to avoid importing training/testing scripts (which may pull in
heavy optional deps like sklearn/matplotlib) in production inference deployments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    """Simple MLP classifier used by miRspa checkpoints."""

    def __init__(
        self,
        input_dim: int = 5,
        max_seq_len: int = 181,
        hidden_dims: list[int] | None = None,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.flatten_dim = input_dim * max_seq_len
        layers: list[nn.Module] = []
        prev_dim = self.flatten_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.mlp(x)

