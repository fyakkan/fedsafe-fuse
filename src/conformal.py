"""Split-conformal calibration utilities for spatially-aware uncertainty (SACP).

Round-2 rewrite. The non-conformity score is the per-pixel residual
``|f(x) - y|`` against a *held-out* target ``y`` (e.g. FLAIR predicted from
T1/T2), which is not observable at inference time.

Two guarantee levels are provided, and they must not be conflated:

* **Patient-level (exact).** The exchangeable unit is the patient. For each
  calibration patient and region we take the (1 - beta) within-patient
  quantile of its pixel residuals; the conformal quantile of these
  per-patient scores, ``tau``, satisfies, for a new exchangeable patient,

      P( fraction of the patient's region pixels with |f - y| <= tau >= 1 - beta ) >= 1 - alpha.

  The guarantee is per region (a union bound gives a joint statement).

* **Pixel-level Mondrian (approximate).** Per-region conformal quantiles over
  pooled calibration pixels. Pixels within a patient are dependent, so the
  nominal pixel coverage is *not* guaranteed; it is validated empirically.

Further tools:

* ``federated_conformal_quantile`` computes the exact pooled conformal quantile
  across clients that only ever release integer counts (one per round,
  compatible with secure aggregation), never scores.
* ``crc_lambda`` implements conformal risk control (Angelopoulos et al.) for a
  monotone bounded loss, e.g. the fraction of tumour-boundary pixels that fall
  outside the interval.

All interval half-widths returned here are ``tau`` (the interval is
``[f - tau, f + tau]``); widths are never reported as ``2 * tau``.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

REGION_BACKGROUND = 0
REGION_INTERIOR = 1
REGION_BOUNDARY = 2
REGION_NAMES = {REGION_BACKGROUND: "background", REGION_INTERIOR: "interior", REGION_BOUNDARY: "boundary"}


# ----------------------------------------------------------------------
# Core quantile
# ----------------------------------------------------------------------
def conformal_rank(n: int, alpha: float) -> int:
    """1-based rank k = ceil((n + 1)(1 - alpha)) of the split-conformal quantile."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    # Guard against float error pushing an exact integer up by one.
    return int(math.ceil((n + 1) * (1.0 - alpha) - 1e-9))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile: the k-th smallest score, k = ceil((n+1)(1-alpha)).

    Returns +inf when k > n (too few calibration points for this alpha).
    Unlike ``np.quantile`` this is the order statistic that carries the
    finite-sample guarantee P(s_{n+1} <= q) >= 1 - alpha.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    n = s.size
    if n == 0:
        raise ValueError("empty calibration set")
    k = conformal_rank(n, alpha)
    if k > n:
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])


def within_unit_quantile(values: np.ndarray, beta: float) -> float:
    """Smallest t such that at least a (1 - beta) fraction of ``values`` is <= t."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return float("nan")
    k = max(1, int(math.ceil((1.0 - beta) * v.size - 1e-9)))
    return float(np.partition(v, k - 1)[k - 1])


# ----------------------------------------------------------------------
# Region masks
# ----------------------------------------------------------------------
def region_map(
    guide: np.ndarray,
    brain_mask: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 100,
    dilate_iter: int = 1,
) -> np.ndarray:
    """Label each pixel background (0) / tissue interior (1) / boundary (2).

    ``guide`` must be computable at inference (e.g. 0.5 * (T1 + T2)), never the
    held-out target. Boundary = dilated Canny edges of ``guide`` restricted to
    the brain mask, so the air/skull rim is not counted as anatomical boundary.
    """
    import cv2

    u8 = (np.clip(guide, 0.0, 1.0) * 255).astype(np.uint8)
    edges = cv2.Canny(u8, canny_low, canny_high)
    if dilate_iter > 0:
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=dilate_iter)
    brain = brain_mask.astype(bool)
    out = np.full(guide.shape, REGION_BACKGROUND, dtype=np.uint8)
    out[brain] = REGION_INTERIOR
    out[brain & (edges > 0)] = REGION_BOUNDARY
    return out


def region_fractions(labels: Sequence[np.ndarray]) -> dict:
    """Pixel fraction per region, pooled over a list of label maps."""
    counts = np.zeros(3, dtype=np.int64)
    for lab in labels:
        counts += np.bincount(lab.ravel(), minlength=3)[:3]
    tot = counts.sum()
    return {REGION_NAMES[r]: float(counts[r] / tot) for r in range(3)}


# ----------------------------------------------------------------------
# Pixel-level Mondrian (approximate) thresholds
# ----------------------------------------------------------------------
def mondrian_thresholds(
    residuals: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    alpha: float,
    regions: Sequence[int] = (REGION_BACKGROUND, REGION_INTERIOR, REGION_BOUNDARY),
) -> dict:
    """Per-region conformal quantile over pooled calibration pixels (approximate).

    Also returns ``'global'``: one threshold over all pixels, the baseline that
    shows whether spatial splitting improves region-conditional coverage.
    """
    res = np.concatenate([r.ravel() for r in residuals])
    lab = np.concatenate([l.ravel() for l in labels])
    out = {"global": conformal_quantile(res, alpha)}
    for r in regions:
        sel = res[lab == r]
        out[REGION_NAMES[r]] = conformal_quantile(sel, alpha) if sel.size else float("nan")
    return out


def pixel_coverage(
    residuals: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    thresholds: dict,
    use_global: bool = False,
) -> dict:
    """Per-region pooled pixel coverage of test residuals under ``thresholds``."""
    res = np.concatenate([r.ravel() for r in residuals])
    lab = np.concatenate([l.ravel() for l in labels])
    out = {}
    for r, name in REGION_NAMES.items():
        sel = res[lab == r]
        if sel.size == 0 or name not in thresholds:
            continue
        tau = thresholds["global"] if use_global else thresholds[name]
        out[name] = float(np.mean(sel <= tau))
    return out


# ----------------------------------------------------------------------
# Patient-level (exact) guarantee
# ----------------------------------------------------------------------
def patient_scores(
    residuals_by_patient: Sequence[Sequence[np.ndarray]],
    labels_by_patient: Sequence[Sequence[np.ndarray]],
    beta: float,
    region: int | None,
) -> np.ndarray:
    """One score per patient: (1 - beta) within-patient quantile of region residuals.

    ``region=None`` pools all pixels (the global baseline). Patients with no
    pixels in the region get NaN and should be dropped by the caller.
    """
    out = np.empty(len(residuals_by_patient), dtype=np.float64)
    for i, (res_list, lab_list) in enumerate(zip(residuals_by_patient, labels_by_patient)):
        res = np.concatenate([r.ravel() for r in res_list])
        if region is not None:
            lab = np.concatenate([l.ravel() for l in lab_list])
            res = res[lab == region]
        out[i] = within_unit_quantile(res, beta)
    return out


def patient_level_threshold(scores: np.ndarray, alpha: float) -> float:
    """Exact patient-level threshold: conformal quantile of per-patient scores."""
    s = np.asarray(scores, dtype=np.float64)
    return conformal_quantile(s[~np.isnan(s)], alpha)


def patient_coverage_indicator(test_scores: np.ndarray, tau: float) -> np.ndarray:
    """1 where a test patient's (1 - beta) region fraction is covered at ``tau``."""
    s = np.asarray(test_scores, dtype=np.float64)
    s = s[~np.isnan(s)]
    return (s <= tau).astype(np.int64)


# ----------------------------------------------------------------------
# Federated exact quantile (counts only)
# ----------------------------------------------------------------------
def _f64_to_ordered_int(x: float) -> int:
    """Map a non-negative float64 to an int with the same ordering."""
    return int(np.array(x, dtype=np.float64).view(np.int64))


def _ordered_int_to_f64(i: int) -> float:
    return float(np.array(i, dtype=np.int64).view(np.float64))


def federated_conformal_quantile(
    client_count_fns: Sequence[Callable[[float], int]],
    client_sizes: Sequence[int],
    alpha: float,
    upper: float,
    return_rounds: bool = False,
):
    """Exact pooled split-conformal quantile from count queries only.

    Each client k exposes ``count_k(t) = #{its scores <= t}`` and its size n_k
    (both integers; with secure aggregation the server only sees the sums).
    The server binary-searches the IEEE-754 bit pattern of t, which is
    order-preserving for non-negative floats, so after at most 64 rounds the
    result is *bit-identical* to ``conformal_quantile`` on the pooled scores.
    Scores must be non-negative and <= ``upper``. No score is ever released.
    """
    n = int(sum(client_sizes))
    k = conformal_rank(n, alpha)
    if k > n:
        return (float("inf"), 0) if return_rounds else float("inf")

    def total(t: float) -> int:
        return int(sum(fn(t) for fn in client_count_fns))

    lo, hi = -1, _f64_to_ordered_int(float(upper))  # count(lo) < k <= count(hi)
    if total(_ordered_int_to_f64(hi)) < k:
        raise ValueError("upper is below the k-th pooled score")
    rounds = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        rounds += 1
        if total(_ordered_int_to_f64(mid)) >= k:
            hi = mid
        else:
            lo = mid
    q = _ordered_int_to_f64(hi)
    return (q, rounds) if return_rounds else q


def count_fn(scores: np.ndarray) -> Callable[[float], int]:
    """Client-side count oracle over a private score array."""
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    return lambda t: int(np.searchsorted(s, t, side="right"))


# ----------------------------------------------------------------------
# Conformal risk control
# ----------------------------------------------------------------------
def crc_lambda(loss_matrix: np.ndarray, lambdas: np.ndarray, alpha: float, B: float = 1.0) -> float:
    """Conformal risk control (Angelopoulos, Bates, Fisch, Lei, Schuster).

    ``loss_matrix[i, j]`` is unit i's loss at ``lambdas[j]``; losses must be
    non-increasing in lambda and bounded in [0, B]. Returns the smallest
    lambda with (n / (n + 1)) * mean_i L_i(lambda) + B / (n + 1) <= alpha,
    which guarantees E[L_{n+1}(lambda_hat)] <= alpha. Returns +inf if no grid
    value qualifies. Federated use: clients send per-lambda loss sums and n_k.
    """
    L = np.asarray(loss_matrix, dtype=np.float64)
    lam = np.asarray(lambdas, dtype=np.float64)
    if np.any(np.diff(lam) <= 0):
        raise ValueError("lambdas must be strictly increasing")
    return crc_from_sums(L.sum(axis=0), L.shape[0], lam, alpha, B)


def crc_from_sums(loss_sums: np.ndarray, n: int, lambdas: np.ndarray, alpha: float, B: float = 1.0) -> float:
    """CRC from aggregated per-lambda loss sums (the federated form)."""
    bound = (np.asarray(loss_sums, dtype=np.float64) + B) / (n + 1)
    ok = np.nonzero(bound <= alpha)[0]
    return float(lambdas[ok[0]]) if ok.size else float("inf")


def miscoverage_loss_curve(
    residual: np.ndarray, scale: np.ndarray, mask: np.ndarray, lambdas: np.ndarray
) -> np.ndarray:
    """Fraction of ``mask`` pixels with residual > lambda * scale, for each lambda.

    With ``mask`` = tumour-boundary pixels this is the CRC loss; ``scale`` is
    the per-pixel interval shape (e.g. the Mondrian regional threshold map),
    which is available at inference, while ``mask`` is needed only for
    calibration. Returns NaN-filled curve if the mask is empty.
    """
    m = mask.astype(bool)
    if not m.any():
        return np.full(len(lambdas), np.nan)
    ratio = np.sort(residual[m] / np.maximum(scale[m], 1e-12))
    covered = np.searchsorted(ratio, lambdas, side="right")
    return 1.0 - covered / ratio.size


# ----------------------------------------------------------------------
# Intervals
# ----------------------------------------------------------------------
def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple:
    """Exact two-sided binomial CI for k successes out of n."""
    from scipy.stats import beta as beta_dist

    a = 1.0 - conf
    lo = 0.0 if k == 0 else float(beta_dist.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def cluster_bootstrap_coverage(
    covered: np.ndarray, totals: np.ndarray, n_boot: int = 2000, conf: float = 0.95, seed: int = 0
) -> tuple:
    """Patient-cluster bootstrap CI for pooled pixel coverage.

    ``covered[i]`` / ``totals[i]`` are patient i's covered and total pixels.
    Resampling whole patients respects within-patient pixel dependence.
    """
    c = np.asarray(covered, dtype=np.float64)
    t = np.asarray(totals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, c.size, size=(n_boot, c.size))
    stats = c[idx].sum(axis=1) / t[idx].sum(axis=1)
    a = (1.0 - conf) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))
