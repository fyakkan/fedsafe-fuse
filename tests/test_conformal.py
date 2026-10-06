"""Unit tests for src/conformal.py. Run: python3 -m pytest tests -q"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import conformal as cp  # noqa: E402


def test_conformal_rank_and_quantile_order_statistic():
    s = np.arange(1, 20, dtype=float)  # n = 19
    # alpha = 0.1 -> k = ceil(20 * 0.9) = 18
    assert cp.conformal_rank(19, 0.1) == 18
    assert cp.conformal_quantile(s, 0.1) == 18.0
    # k > n -> +inf
    assert math.isinf(cp.conformal_quantile(np.arange(5.0), 0.1))
    # differs from interpolated np.quantile
    assert cp.conformal_quantile(s, 0.1) != np.quantile(s, 0.9)


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_marginal_coverage_monte_carlo(alpha):
    rng = np.random.default_rng(0)
    n, reps = 50, 20000
    cal = rng.exponential(size=(reps, n))
    test = rng.exponential(size=reps)
    k = cp.conformal_rank(n, alpha)
    q = np.sort(cal, axis=1)[:, k - 1]
    cov = np.mean(test <= q)
    lo, hi = 1 - alpha, k / (n + 1)  # exact expectation is k/(n+1) for continuous scores
    se = math.sqrt(hi * (1 - hi) / reps)
    assert lo - 4 * se <= cov <= hi + 4 * se


def test_within_unit_quantile():
    v = np.arange(1, 11, dtype=float)
    assert cp.within_unit_quantile(v, 0.1) == 9.0
    assert np.mean(v <= cp.within_unit_quantile(v, 0.1)) >= 0.9


def test_patient_level_guarantee_with_dependent_pixels():
    """Pixels within a patient share a random scale (strong dependence); the
    patient-level statement must still hold at >= 1 - alpha."""
    rng = np.random.default_rng(1)
    alpha, beta, n_cal, reps = 0.1, 0.1, 30, 3000
    hits = 0
    for _ in range(reps):
        def patient():
            scale = rng.lognormal(0, 0.5)
            return [np.abs(rng.normal(0, scale, size=(8, 8)))]
        res = [patient() for _ in range(n_cal + 1)]
        lab = [[np.ones((8, 8), np.uint8)] for _ in range(n_cal + 1)]
        scores = cp.patient_scores(res, lab, beta, region=cp.REGION_INTERIOR)
        tau = cp.patient_level_threshold(scores[:n_cal], alpha)
        frac = np.mean(res[-1][0] <= tau)
        hits += frac >= 1 - beta
    cov = hits / reps
    assert cov >= 1 - alpha - 4 * math.sqrt(alpha * (1 - alpha) / reps)


def test_region_map_excludes_background_from_boundary():
    img = np.zeros((64, 64), np.float32)
    img[16:48, 16:48] = 0.8
    img[28:36, 28:36] = 0.2  # internal edge
    brain = img > 0
    lab = cp.region_map(img, brain)
    assert np.all(lab[~brain] == cp.REGION_BACKGROUND)
    assert (lab == cp.REGION_BOUNDARY).any() and (lab == cp.REGION_INTERIOR).any()
    fr = cp.region_fractions([lab])
    assert abs(sum(fr.values()) - 1) < 1e-12


def test_mondrian_and_global_thresholds():
    rng = np.random.default_rng(2)
    lab = rng.integers(0, 3, size=(20, 16, 16)).astype(np.uint8)
    res = np.abs(rng.normal(size=lab.shape)) * (1 + lab)  # boundary noisier
    th = cp.mondrian_thresholds(list(res), list(lab), 0.1)
    assert th["background"] < th["interior"] < th["boundary"]
    cov_m = cp.pixel_coverage(list(res), list(lab), th)
    cov_g = cp.pixel_coverage(list(res), list(lab), th, use_global=True)
    assert all(v >= 0.89 for v in cov_m.values())
    assert cov_g["boundary"] < cov_m["boundary"]  # global under-covers the noisy region


def test_federated_quantile_bit_identical():
    rng = np.random.default_rng(3)
    clients = [rng.gamma(2.0, 0.05, size=n) for n in (37, 112, 9)]
    pooled = np.concatenate(clients)
    for alpha in (0.05, 0.1, 0.15):
        q_c = cp.conformal_quantile(pooled, alpha)
        q_f, rounds = cp.federated_conformal_quantile(
            [cp.count_fn(c) for c in clients], [c.size for c in clients], alpha, upper=10.0, return_rounds=True
        )
        assert q_f == q_c
        assert rounds <= 64
    # k > n -> inf
    assert math.isinf(cp.federated_conformal_quantile([cp.count_fn(np.ones(3))], [3], 0.1, upper=1.0))


def test_crc_controls_risk():
    rng = np.random.default_rng(4)
    alpha, n, reps = 0.1, 40, 2000
    lambdas = np.linspace(0, 5, 501)
    risks = []
    for _ in range(reps):
        curves = []
        for _ in range(n + 1):
            r = np.abs(rng.normal(0, rng.lognormal(0, 0.3), size=(16, 16)))
            mask = np.zeros((16, 16), bool)
            mask[4:12, 4:12] = True
            curves.append(cp.miscoverage_loss_curve(r, np.ones_like(r), mask, lambdas))
        curves = np.array(curves)
        lam = cp.crc_lambda(curves[:n], lambdas, alpha)
        j = np.searchsorted(lambdas, lam)
        risks.append(curves[n, j])
    assert np.mean(risks) <= alpha + 3 * np.std(risks) / math.sqrt(reps)


def test_crc_federated_equals_central():
    rng = np.random.default_rng(5)
    lambdas = np.linspace(0, 1, 101)
    L = np.clip(1 - lambdas[None, :] * rng.uniform(0.5, 2, size=(30, 1)), 0, 1)
    central = cp.crc_lambda(L, lambdas, 0.2)
    sums = L[:10].sum(0) + L[10:].sum(0)
    assert cp.crc_from_sums(sums, 30, lambdas, 0.2) == central


def test_clopper_pearson_and_bootstrap():
    lo, hi = cp.clopper_pearson(90, 100)
    assert lo < 0.9 < hi
    assert cp.clopper_pearson(0, 10)[0] == 0.0 and cp.clopper_pearson(10, 10)[1] == 1.0
    lo, hi = cp.cluster_bootstrap_coverage(np.full(30, 90), np.full(30, 100))
    assert lo == hi == 0.9
