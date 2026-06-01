"""In-process FedAvg simulator supporting standard, DP-SGD, and FIPCA modes.

A single-process replacement for Flower's SimulationEngine, designed for Colab
Free's compute budget. Mirrors the round structure of Flower so the algorithm
can be swapped to Flower without changing the experimental protocol.

Three modes:
    - 'standard': vanilla FedAvg (raw weight deltas).
    - 'dpsgd'   : DP-SGD on each client (per-batch gradient clip + Gaussian noise),
                  then standard FedAvg aggregation of resulting weight deltas.
    - 'fipca'   : Client weight delta is projected to a rank-d subspace via a
                  pseudo-random Gaussian matrix derived from a fixed seed (shared
                  basis across clients, never transmitted). Server averages the
                  rank-d coefficients and reconstructs.

Communication cost per round is logged in bytes, which is the primary privacy /
efficiency claim of FedSafe-Fuse.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


def _flatten_state(state: dict) -> torch.Tensor:
    """Concatenate all tensors in a state_dict into a single 1-D float32 vector."""
    return torch.cat([v.flatten().float() for v in state.values()])


def _unflatten_to_state(flat: torch.Tensor, template: dict) -> dict:
    """Inverse of _flatten_state, using template state_dict for shapes."""
    out: dict = {}
    idx = 0
    for k, v in template.items():
        n = v.numel()
        out[k] = flat[idx : idx + n].reshape(v.shape).to(v.dtype)
        idx += n
    return out


class FedAvgTrainer:
    """Single-process federated trainer.

    Parameters
    ----------
    build_model_fn :
        Zero-arg callable that returns a fresh model on CPU. Cloned per client.
    loss_fn_factory :
        Zero-arg callable returning a loss module (e.g. FusionLoss(beta)).
        Recreated per client to avoid sharing buffers (SSIM Gaussian window).
    client_datasets :
        List of K Dataset objects, one per client. Items must be (mri, pet, ...).
    eval_dataset :
        Held-out evaluation set (e.g. the calibration split).
    device :
        'cuda' or 'cpu'.
    mode :
        'standard', 'dpsgd', or 'fipca'.
    samples_per_local_epoch :
        Cap on examples drawn per client per local epoch (Round 1 compute cut).
    fipca_rank :
        d, the rank of the FIPCA random-projection subspace.
    dp_clip, dp_sigma :
        DP-SGD per-batch gradient L2-clip threshold and noise multiplier.
    lr :
        Adam learning rate.
    batch_size :
        Per-client batch size.
    seed :
        Random seed for reproducibility.
    """

    VALID_MODES = ("standard", "dpsgd", "fipca")

    def __init__(
        self,
        build_model_fn: Callable[[], nn.Module],
        loss_fn_factory: Callable[[], nn.Module],
        client_datasets: list,
        eval_dataset: Dataset,
        device: str = "cuda",
        mode: str = "standard",
        samples_per_local_epoch: int = 100,
        fipca_rank: int = 32,
        dp_clip: float = 1.0,
        dp_sigma: float = 0.5,
        lr: float = 1e-3,
        batch_size: int = 16,
        seed: int = 42,
    ):
        assert mode in self.VALID_MODES, f"mode must be one of {self.VALID_MODES}"
        self.build_model_fn = build_model_fn
        self.loss_fn_factory = loss_fn_factory
        self.client_datasets = client_datasets
        self.eval_dataset = eval_dataset
        self.device = device
        self.mode = mode
        self.samples_per_local_epoch = samples_per_local_epoch
        self.fipca_rank = fipca_rank
        self.dp_clip = dp_clip
        self.dp_sigma = dp_sigma
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Global model (server's authoritative copy)
        self.global_model = build_model_fn().to(device)
        self.global_state = {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}

        # FIPCA: deterministic ORTHONORMAL projection matrix P in (rank, n_params).
        # Built via QR decomposition so that P @ P.T = I_rank exactly, which makes
        # P.T @ P a true rank-d orthogonal projector with operator norm 1.
        # Earlier scaled-Gaussian attempt had operator norm ~sqrt(n_params/rank),
        # causing reconstructed deltas to blow up and Adam to diverge (NaN).
        if mode == "fipca":
            n_params = sum(v.numel() for v in self.global_state.values())
            gen = torch.Generator(device=device).manual_seed(seed)
            G = torch.randn(n_params, fipca_rank, device=device, generator=gen)
            Q, _ = torch.linalg.qr(G)  # Q has orthonormal columns: (n_params, rank)
            self.P = Q.T.contiguous()  # (rank, n_params) with orthonormal rows

    # ------------------------------------------------------------------
    # Client-side local training
    # ------------------------------------------------------------------
    def _local_subset_loader(self, client_idx: int) -> DataLoader:
        ds = self.client_datasets[client_idx]
        n = min(self.samples_per_local_epoch, len(ds))
        sel = self._rng.choice(len(ds), size=n, replace=False)
        subset = Subset(ds, sel.tolist())
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def _train_client(self, client_idx: int, n_local_epochs: int) -> dict:
        """Run E local epochs on a fresh random subset of this client's data, return weight delta."""
        local_model = self.build_model_fn().to(self.device)
        local_model.load_state_dict(self.global_state)
        opt = torch.optim.Adam(local_model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = self.loss_fn_factory().to(self.device)

        local_model.train()
        for _ in range(n_local_epochs):
            loader = self._local_subset_loader(client_idx)
            for batch in loader:
                mri, pet = batch[0].to(self.device), batch[1].to(self.device)
                fused = local_model(mri, pet)
                loss = loss_fn(fused, mri, pet)
                opt.zero_grad()
                loss.backward()
                if self.mode == "dpsgd":
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), self.dp_clip)
                    for p in local_model.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad) * self.dp_sigma * self.dp_clip)
                opt.step()

        local_state = local_model.state_dict()
        delta = {k: (local_state[k].detach() - self.global_state[k]).to(self.device) for k in self.global_state}
        return delta

    # ------------------------------------------------------------------
    # Server-side aggregation
    # ------------------------------------------------------------------
    def _compress_delta(self, delta: dict) -> tuple:
        """Return (payload, bytes_sent) for one client's update."""
        if self.mode == "fipca":
            flat = _flatten_state(delta).to(self.device)
            coeffs = self.P @ flat  # (rank,)
            payload = coeffs.detach().clone()
            bytes_sent = payload.numel() * 4  # float32
            return payload, bytes_sent
        else:
            payload = {k: v.detach().clone() for k, v in delta.items()}
            bytes_sent = sum(v.numel() * 4 for v in payload.values())
            return payload, bytes_sent

    def _aggregate(self, client_payloads: list) -> dict:
        if self.mode == "fipca":
            avg_coeffs = torch.stack(client_payloads).mean(dim=0)  # (rank,)
            recon_flat = self.P.T @ avg_coeffs  # (n_params,)
            return _unflatten_to_state(recon_flat, self.global_state)
        else:
            avg = {}
            for k in self.global_state:
                stacked = torch.stack([cp[k] for cp in client_payloads])
                if stacked.dtype.is_floating_point or stacked.dtype.is_complex:
                    avg[k] = stacked.mean(dim=0)
                else:
                    # Integer buffers (e.g. BatchNorm.num_batches_tracked) cannot be
                    # mean()-ed directly; promote to float, mean, and cast back.
                    avg[k] = stacked.to(torch.float32).mean(dim=0).to(stacked.dtype)
            return avg

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, ssim_score_fn, psnr_score_fn) -> dict:
        self.global_model.load_state_dict(self.global_state)
        self.global_model.eval()
        loss_fn = self.loss_fn_factory().to(self.device)
        losses, ssims, psnrs = [], [], []
        loader = DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        for batch in loader:
            mri, pet = batch[0].to(self.device), batch[1].to(self.device)
            fused = self.global_model(mri, pet)
            losses.append(loss_fn(fused, mri, pet).item())
            ref = 0.5 * (mri + pet)
            ssims.append(ssim_score_fn(fused, ref))
            psnrs.append(psnr_score_fn(fused, ref))
        return {
            "eval_loss": float(np.mean(losses)),
            "eval_ssim": float(np.mean(ssims)),
            "eval_psnr": float(np.mean(psnrs)),
        }

    # ------------------------------------------------------------------
    # Full federated run
    # ------------------------------------------------------------------
    def run(
        self,
        T: int,
        E: int,
        ssim_score_fn,
        psnr_score_fn,
        log_every: int = 5,
        checkpoint_every: int = 10,
        checkpoint_path: Optional[str] = None,
        tag: str = "run",
    ) -> list:
        K = len(self.client_datasets)
        history = []
        for t in range(1, T + 1):
            t0 = time.time()
            payloads, total_bytes = [], 0
            for k in range(K):
                delta = self._train_client(k, E)
                payload, bytes_sent = self._compress_delta(delta)
                payloads.append(payload)
                total_bytes += bytes_sent
            avg_delta = self._aggregate(payloads)
            for k in self.global_state:
                self.global_state[k] = self.global_state[k] + avg_delta[k]
            self.global_model.load_state_dict(self.global_state)

            row = {
                "tag": tag,
                "round": t,
                "mode": self.mode,
                "round_secs": time.time() - t0,
                "bytes_per_round": int(total_bytes),
                "samples_per_local_epoch": self.samples_per_local_epoch,
                "fipca_rank": self.fipca_rank if self.mode == "fipca" else None,
                "dp_sigma": self.dp_sigma if self.mode == "dpsgd" else None,
            }
            if t % log_every == 0 or t == 1 or t == T:
                row.update(self.evaluate(ssim_score_fn, psnr_score_fn))
                print(
                    f"  [{tag}] r{t:02d}/{T} | "
                    f"loss={row['eval_loss']:.4f} ssim={row['eval_ssim']:.4f} "
                    f"psnr={row['eval_psnr']:.2f}dB | "
                    f"bytes={row['bytes_per_round']/1e6:.1f}MB/rd | "
                    f"{row['round_secs']:.1f}s"
                )
            if checkpoint_path and (t % checkpoint_every == 0 or t == T):
                torch.save(
                    {"state_dict": self.global_state, "round": t, "tag": tag},
                    f"{checkpoint_path}_r{t}.pt",
                )
            history.append(row)
        return history
