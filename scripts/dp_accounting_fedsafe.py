"""Formal (epsilon, delta)-DP accounting for the FedSafe-Fuse DP-SGD baseline.

Round-2 reviewer ask #4: "Add the formal (epsilon, delta)-DP accounting promised
for the DP-SGD baseline to make the privacy comparison analytical, not only
empirical."

We use the Renyi Differential Privacy (RDP) accountant for the Sampled Gaussian
Mechanism (SGM):
  * RDP of the SGM at integer orders: Mironov, Talwar, Zhang (2019),
    "Renyi Differential Privacy of the Sampled Gaussian Mechanism", Eq. (5)-(6).
  * Composition: RDP is additive over the n training steps (Mironov 2017).
  * RDP -> (eps, delta) conversion: Canonne, Kamath, Steinke (2020) tightened
    bound (also in Mironov 2017 in looser form).

No external DP library required (no opacus/dp_accounting); only numpy + scipy,
so it reproduces locally and in CI. Validated against the closed-form
non-subsampled Gaussian RDP (q=1 -> alpha/(2 sigma^2)).
"""
from __future__ import annotations

import csv
import math
import os
from typing import Iterable

import numpy as np
from scipy import special


def _log_add(logx: float, logy: float) -> float:
    """Numerically stable log(exp(logx) + exp(logy))."""
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:
        return b
    return b + math.log1p(math.exp(a - b))


def _log_comb(n: int, k: int) -> float:
    return (special.gammaln(n + 1) - special.gammaln(k + 1)
            - special.gammaln(n - k + 1))


def _log_erfc(x: float) -> float:
    return math.log(2) + special.log_ndtr(-x * math.sqrt(2))


def rdp_sgm_int(q: float, sigma: float, alpha: int) -> float:
    """RDP at *integer* order alpha for one step of the Sampled Gaussian Mechanism.

    Mironov, Talwar, Zhang (2019), binomial expansion. Returns RDP epsilon at
    this order for a SINGLE application of the mechanism.
    """
    if q == 0.0:
        return 0.0
    if q == 1.0:                       # no subsampling: closed form
        return alpha / (2.0 * sigma ** 2)
    # log of sum_{k=0}^{alpha} C(alpha,k) (1-q)^(alpha-k) q^k exp(k(k-1)/(2 sigma^2))
    log_a = -np.inf
    for k in range(alpha + 1):
        log_term = (_log_comb(alpha, k)
                    + (alpha - k) * math.log1p(-q)
                    + k * math.log(q)
                    + (k * (k - 1)) / (2.0 * sigma ** 2))
        log_a = _log_add(log_a, log_term)
    return log_a / (alpha - 1)


def compute_rdp(q: float, sigma: float, steps: int,
                orders: Iterable[float]) -> np.ndarray:
    """Total RDP across `steps` compositions, evaluated at each order."""
    return np.array([steps * rdp_sgm_int(q, sigma, int(a)) for a in orders])


def rdp_to_eps(orders: np.ndarray, rdp: np.ndarray, delta: float):
    """Convert RDP curve to (eps, delta) using the tightened CKS-2020 bound.

    Returns (eps, best_order).
    """
    orders = np.asarray(orders, dtype=float)
    rdp = np.asarray(rdp, dtype=float)
    # Canonne-Kamath-Steinke 2020, Prop. 12 (as implemented in Google DP):
    eps = (rdp
           - (np.log(delta) + np.log(orders)) / (orders - 1)
           + np.log1p(-1.0 / orders))
    idx = int(np.nanargmin(eps))
    return float(eps[idx]), float(orders[idx])


def dp_sgd_epsilon(q: float, sigma: float, steps: int, delta: float):
    orders = np.concatenate([np.arange(2, 64), np.arange(64, 512, 4),
                             np.arange(512, 4096, 32)])
    rdp = compute_rdp(q, sigma, steps, orders)
    return rdp_to_eps(orders, rdp, delta)


def _self_test():
    # q=1 reduces to closed-form non-subsampled Gaussian RDP alpha/(2 sigma^2).
    for sigma in (0.5, 1.0, 2.0):
        for a in (2, 5, 32):
            got = rdp_sgm_int(1.0, sigma, a)
            exp = a / (2 * sigma ** 2)
            assert abs(got - exp) < 1e-9, (sigma, a, got, exp)
    # Monotone in steps, and small-q amplification: eps(q) << eps(1) at same steps.
    e_small, _ = dp_sgd_epsilon(0.001, 1.0, 1000, 1e-5)
    e_big, _ = dp_sgd_epsilon(0.5, 1.0, 1000, 1e-5)
    assert e_small < e_big
    print("self-test OK (q=1 closed form; subsampling amplification monotone)")


if __name__ == "__main__":
    _self_test()

    # ---- FedSafe-Fuse DP-SGD baseline configuration (Round 1) ----
    sigma = 0.5            # noise multiplier (dp_sigma)
    C = 1.0               # L2 clip (dp_clip); cancels in (eps, delta) accounting
    batch = 16
    samples_per_local_epoch = 40
    E = 5                 # local epochs / round
    T = 50                # communication rounds
    delta = 1e-5
    client_sizes = {"client0": 7714, "client1": 15445, "client2": 6218}

    # batches per local epoch (last batch partial): ceil(40/16) = 3
    batches_per_epoch = math.ceil(samples_per_local_epoch / batch)
    steps = T * E * batches_per_epoch          # noisy SGD steps per client
    print(f"\nDP-SGD config: sigma={sigma}, C={C}, batch={batch}, "
          f"{samples_per_local_epoch} samples/epoch -> {batches_per_epoch} steps/epoch")
    print(f"Total noisy steps per client: T*E*batches = {T}*{E}*{batches_per_epoch} = {steps}")
    print(f"delta = {delta}\n")

    print(f"{'client':9s} {'N':>7s} {'q=batch/N':>11s} {'epsilon':>12s} {'order a*':>9s}")
    worst = None
    csv_rows = []
    for name, N in client_sizes.items():
        q = batch / N
        eps, a = dp_sgd_epsilon(q, sigma, steps, delta)
        print(f"{name:9s} {N:>7d} {q:>11.5f} {eps:>12.1f} {a:>9.0f}")
        csv_rows.append({"client": name, "N": N, "sigma": sigma, "clip": C,
                         "batch": batch, "steps": steps, "q": round(q, 6),
                         "delta": delta, "epsilon": round(eps, 2),
                         "best_order": int(a)})
        if worst is None or eps > worst[1]:
            worst = (name, eps, q, a)

    out_csv = os.path.join(os.path.dirname(__file__), os.pardir,
                           "results", "dp_accounting.csv")
    out_csv = os.path.normpath(out_csv)
    if os.path.isdir(os.path.dirname(out_csv)):
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\nSaved {out_csv}")
    print(f"\nWorst-case client (smallest N => largest q): {worst[0]}, "
          f"epsilon = {worst[1]:.1f} at delta = {delta}")
    print("\nInterpretation: heavy subsampling amplification (q ~ 0.002, only 750 steps)")
    print("keeps the analytical guarantee bounded at eps ~ 6.2 (delta=1e-5), but this is")
    print("a WEAK guarantee -- sigma=0.5 sits below the sigma~1 regime usually required")
    print("for eps<=1. Crucially, buying even this weak formal eps cost 59 MB/round AND")
    print("collapsed utility (SSIM stuck at 0.23). FIPCA attains the SAME empirical DLG")
    print("protection (reconstruction SSIM<=0.013) at 384 B/round, so the privacy")
    print("comparison now holds both empirically (DLG) and analytically (this eps).")
    print("NOTE: fixed-size minibatches are accounted as Poisson subsampling at")
    print("q=batch/N (the standard opacus/TF-Privacy approximation), disclosed in-paper.")
