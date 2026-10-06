"""Regression tests for the Round-2 trainer fixes. Run: python3 -m pytest tests -q"""

import os
import sys
import warnings

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.fed.trainer import FedAvgTrainer  # noqa: E402
from src.losses import TargetLoss, psnr_score, ssim_score  # noqa: E402


class TinyFuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.out = nn.Conv2d(4, 1, 1)

    def forward(self, a, b):
        return torch.sigmoid(self.out(torch.relu(self.bn(self.conv(torch.cat([a, b], 1))))))


def _datasets(n=24, k=3):
    torch.manual_seed(0)
    mk = lambda: TensorDataset(torch.rand(n, 1, 16, 16) + 2.0, torch.rand(n, 1, 16, 16), torch.rand(n, 1, 16, 16))
    return [mk() for _ in range(k)], mk()


def _trainer(mode, **kw):
    clients, ev = _datasets()
    torch.manual_seed(0)
    return FedAvgTrainer(
        TinyFuser, lambda: TargetLoss(), clients, ev, device="cpu", mode=mode,
        samples_per_local_epoch=16, batch_size=8, supervised_target=True, **kw,
    )


def test_frop_updates_bn_buffers():
    """Old bug: BN running stats were projected through FROP and stayed at init."""
    tr = _trainer("fipca", fipca_rank=4)
    tr.run(T=1, E=1, ssim_score_fn=ssim_score, psnr_score_fn=psnr_score, log_every=1)
    # Inputs are shifted by +2, so a real BN update moves running_mean well away from 0.
    assert tr.global_state["bn.running_mean"].abs().max() > 0.05
    assert int(tr.global_state["bn.num_batches_tracked"]) > 0


def test_frop_projects_parameters_only_and_counts_buffer_bytes():
    tr = _trainer("fipca", fipca_rank=4)
    n_params = sum(p.numel() for p in tr.global_model.parameters())
    assert tr.P.shape == (4, n_params)
    hist = tr.run(T=1, E=1, ssim_score_fn=ssim_score, psnr_score_fn=psnr_score, log_every=1)
    buf_bytes = sum(b.numel() * b.element_size() for b in tr.global_model.buffers())
    assert hist[0]["bytes_per_round"] == 3 * (4 * 4 + buf_bytes)


def test_standard_mode_supervised_eval_uses_target():
    tr = _trainer("standard")
    hist = tr.run(T=1, E=1, ssim_score_fn=ssim_score, psnr_score_fn=psnr_score, log_every=1)
    assert "eval_ssim" in hist[0]


def test_dpsgd_mode_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _trainer("dpsgd")
    assert any("NOT DP-SGD" in str(x.message) for x in w)
