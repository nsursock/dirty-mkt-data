"""P0 — stylized-facts validation framework.

- Correctness: the ACF kernels recover known structure (strong lag-1 ACF for
  a linear trend; ~0 for white noise; skewness/kurtosis of known shapes).
- Fit-for-purpose: positive control (sine-modulated vol yields strong
  vol-clustering ACF), and GBM satisfies its own contracts (no clustering,
  no kurtosis, no regime persistence).
- Batch shapes: all stats work on both (T,) and (n_paths, T).
"""

import mlx.core as mx

from dirty_mkt_data.core.gbm import GBM
from dirty_mkt_data.eval.stylized_facts import (
    autocorr,
    excess_kurtosis,
    regime_persistence,
    skewness,
    vol_clustering_acf,
)

T = 8192
SEED = 3


def _wt(t, seed):
    k0, k1 = mx.random.split(mx.random.key(seed), 2)
    return mx.random.normal((t,), key=k0)


def test_acf_of_linear_trend_is_high():
    x = mx.arange(T, dtype=mx.int32).astype(mx.float32)
    ac = autocorr(x, max_lag=3)
    assert bool(mx.all(ac > 0.999))


def test_acf_of_white_noise_is_near_zero():
    x = _wt(T, SEED)
    ac = autocorr(x, max_lag=8)
    assert bool(mx.all(mx.abs(ac) < 3.5 / (T ** 0.5) + 0.01))


def test_acf_sine_has_known_lag_1():
    # cos wave: lag-1 ACF is exactly cos(2pi * 1 * f); f small => near 1
    t = mx.arange(float(T))
    x = mx.cos(2.0 * mx.pi * t / 64.0)
    ac = autocorr(x, max_lag=1)[0]
    assert abs(float(ac) - float(mx.cos(2.0 * mx.pi / 64.0))) < 0.01


def test_vol_clustering_positive_control():
    # slow envelope x noise => squared-return ACF(1) clearly positive
    t = mx.arange(float(T))
    env = 1.0 + 0.9 * mx.sin(2.0 * mx.pi * t / 256.0)
    r = env * _wt(T, SEED)
    acf = float(vol_clustering_acf(r, lag=1))
    assert acf > 0.15


def test_vol_clustering_white_noise_near_zero():
    r = _wt(T, SEED)
    assert abs(float(vol_clustering_acf(r, lag=1))) < 0.05


def test_excess_kurtosis_gaussian_and_uniform():
    r = _wt(T, SEED)
    assert abs(float(excess_kurtosis(r))) < 0.1
    u = mx.random.uniform(shape=(T,), key=mx.random.key(SEED))
    assert abs(float(excess_kurtosis(u)) + 6.0 / 5.0) < 0.05


def test_skewness_symmetric_and_skewed():
    r = _wt(T, SEED)
    assert abs(float(skewness(r))) < 0.05
    # shifted half-normal: clearly positive skew (mean > median shape)
    pos = mx.abs(_wt(T, SEED)) + 1.0
    assert float(skewness(pos)) > 0.5


def test_gbm_satisfies_null_contracts():
    ds = GBM(mu=0.0, sigma=0.2).sample(T, n_paths=64, key=mx.random.key(SEED))
    r = ds.returns
    assert bool(mx.all(mx.abs(vol_clustering_acf(r, lag=1)) < 0.08))
    assert bool(mx.all(mx.abs(excess_kurtosis(r)) < 0.3))
    pers = regime_persistence(r, n_segments=16)
    assert float(pers) < 0.5


def test_batch_shapes():
    r = GBM(sigma=0.2).sample(T, n_paths=32, key=mx.random.key(SEED)).returns
    assert autocorr(r, max_lag=4).shape == (32, 4)
    assert vol_clustering_acf(r, lag=2).shape == (32,)
    assert excess_kurtosis(r).shape == (32,)
    assert skewness(r).shape == (32,)