"""IFCNN baseline (Zhang et al. 2021).

A general image fusion framework based on a two-stream CNN with element-wise fusion
followed by reconstruction. Simplified from the original paper for our 128x128
single-channel fusion setting. Used as Baseline 2 in the FedSafe-Fuse evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IFCNN(nn.Module):
    """Two-stream conv + element-wise fusion + conv reconstruction."""

    FUSION_MODES = ("max", "mean", "sum")

    def __init__(self, base_ch: int = 64, fusion: str = "max"):
        super().__init__()
        assert fusion in self.FUSION_MODES, f"fusion must be one of {self.FUSION_MODES}"
        self.fusion_mode = fusion
        self.stream1 = self._make_stream(base_ch)
        self.stream2 = self._make_stream(base_ch)
        self.recon = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            nn.BatchNorm2d(base_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 1, kernel_size=1),
        )

    @staticmethod
    def _make_stream(c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        f1 = self.stream1(x1)
        f2 = self.stream2(x2)
        if self.fusion_mode == "max":
            fused = torch.maximum(f1, f2)
        elif self.fusion_mode == "mean":
            fused = 0.5 * (f1 + f2)
        else:
            fused = f1 + f2
        return torch.sigmoid(self.recon(fused))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = IFCNN()
    n = count_parameters(m)
    print(f"IFCNN trainable params: {n:,} ({n / 1e6:.2f}M)")
    x = torch.randn(2, 1, 128, 128)
    y = m(x, x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
