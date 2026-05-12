"""CNN models for image-based story position classification."""

from __future__ import annotations

import torch
from torch import nn


class StoryPositionCNN(nn.Module):
    """Small CNN classifier with configurable single-aspect changes."""

    def __init__(
        self,
        num_classes: int = 5,
        filters: tuple[int, int, int] = (16, 32, 64),
        kernel_size: int = 3,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_channels = 3

        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.extend([nn.ReLU(), nn.MaxPool2d(2)])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(filters[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
