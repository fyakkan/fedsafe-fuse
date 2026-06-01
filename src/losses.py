"""Composite fusion loss: L1 + beta * (1 - SSIM).

Implements a differentiable SSIM via a Gaussian-windowed convolution
(no external dependency on pytorch_msssim). Designed for single-channel
medical images in [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w_2d = g.unsqueeze(0) * g.unsqueeze(1)  # (W, W)
    return w_2d.expand(channels, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    """1 - SSIM, computed with a Gaussian sliding window. Single-channel input in [0, 1]."""

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 1.0,
        channels: int = 1,
    ):
        super().__init__()
        self.window_size = window_size
        self.data_range = data_range
        self.C1 = (0.01 * data_range) ** 2
        self.C2 = (0.03 * data_range) ** 2
        self.register_buffer("window", _gaussian_window(window_size, sigma, channels))

    def _filter(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.window, padding=self.window_size // 2, groups=x.shape[1])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mu_p = self._filter(pred)
        mu_t = self._filter(target)
        mu_p2 = mu_p ** 2
        mu_t2 = mu_t ** 2
        mu_pt = mu_p * mu_t
        sigma_p2 = self._filter(pred * pred) - mu_p2
        sigma_t2 = self._filter(target * target) - mu_t2
        sigma_pt = self._filter(pred * target) - mu_pt
        num = (2 * mu_pt + self.C1) * (2 * sigma_pt + self.C2)
        den = (mu_p2 + mu_t2 + self.C1) * (sigma_p2 + sigma_t2 + self.C2)
        ssim_map = num / den
        return 1.0 - ssim_map.mean()


class FusionLoss(nn.Module):
    """L1(fused, ref) + beta * SSIMLoss(fused, ref).

    Reference composite is a per-pixel weighted average of source modalities, per the proposal.
    Default weights are (0.5, 0.5). Override `reference_weights` per dataset if needed.
    """

    def __init__(self, beta: float = 1.0, reference_weights: tuple[float, float] = (0.5, 0.5)):
        super().__init__()
        assert abs(sum(reference_weights) - 1.0) < 1e-6, "reference_weights must sum to 1.0"
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.beta = beta
        w1, w2 = reference_weights
        self.register_buffer("w1", torch.tensor(w1))
        self.register_buffer("w2", torch.tensor(w2))

    def reference(self, src1: torch.Tensor, src2: torch.Tensor) -> torch.Tensor:
        return self.w1 * src1 + self.w2 * src2

    def forward(
        self,
        fused: torch.Tensor,
        src1: torch.Tensor,
        src2: torch.Tensor,
    ) -> torch.Tensor:
        ref = self.reference(src1, src2)
        return self.l1(fused, ref) + self.beta * self.ssim(fused, ref)


@torch.no_grad()
def ssim_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM (higher is better) for evaluation. Mirrors SSIMLoss without the 1 - ... wrap."""
    loss_fn = SSIMLoss().to(pred.device)
    return float(1.0 - loss_fn(pred, target).item())


@torch.no_grad()
def psnr_score(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR in dB."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(data_range ** 2 / mse)).item()
