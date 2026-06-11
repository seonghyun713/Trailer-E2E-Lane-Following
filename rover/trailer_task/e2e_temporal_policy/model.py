#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrameEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        channels: Sequence[int] = (24, 32, 48, 64),
        feature_dim: int = 128,
    ) -> None:
        super().__init__()
        blocks = []
        current = in_channels
        for out_channels in channels:
            blocks.append(ConvBlock(current, int(out_channels), stride=2))
            current = int(out_channels)
        self.cnn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(current, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.cnn(x)))


class TemporalPolicyNet(nn.Module):
    def __init__(
        self,
        scalar_dim: int,
        num_modes: int,
        in_channels: int = 2,
        cnn_channels: Sequence[int] = (24, 32, 48, 64),
        frame_feature_dim: int = 128,
        scalar_embed_dim: int = 32,
        temporal_hidden_dim: int = 128,
        temporal_layers: int = 1,
        dropout: float = 0.10,
        max_motor: float = 1.0,
    ) -> None:
        super().__init__()
        self.max_motor = float(max_motor)
        self.encoder = FrameEncoder(in_channels, cnn_channels, frame_feature_dim)
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, scalar_embed_dim),
            nn.LayerNorm(scalar_embed_dim),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.GRU(
            input_size=frame_feature_dim + scalar_embed_dim,
            hidden_size=temporal_hidden_dim,
            num_layers=temporal_layers,
            batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        head_in = temporal_hidden_dim
        self.shared_head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_in),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.motor_head = nn.Linear(head_in, 2)
        self.mode_head = nn.Linear(head_in, num_modes)

    def forward(self, bev: torch.Tensor, scalars: torch.Tensor) -> Dict[str, torch.Tensor]:
        if bev.ndim != 5:
            raise ValueError(f"Expected bev as B,T,C,H,W, got {tuple(bev.shape)}")
        batch, steps, channels, height, width = bev.shape
        x = bev.reshape(batch * steps, channels, height, width)
        image_feat = self.encoder(x).reshape(batch, steps, -1)
        scalar_feat = self.scalar_net(scalars.reshape(batch * steps, -1)).reshape(batch, steps, -1)
        temporal_in = torch.cat([image_feat, scalar_feat], dim=-1)
        temporal_out, _ = self.temporal(temporal_in)
        last = temporal_out[:, -1]
        shared = self.shared_head(last)
        wheels = torch.tanh(self.motor_head(shared)) * self.max_motor
        return {
            "wheels": wheels,
            "mode_logits": self.mode_head(shared),
        }

