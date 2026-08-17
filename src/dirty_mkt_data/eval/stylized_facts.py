"""Stylized-facts validation framework (Apple MLX only, fully vectorized).

Every statistic is computed as an MLX tensor over the *last* axis (time),
supporting batch shapes like ``(n_paths, n_steps)`` — no Python loops, no
``.item()`` / ``.tolist()`` escapes in the library.

Tests built on these helpers (see ``tests/test_stylized_facts.py``):

* ``vol_clustering_acf``      — ACF of squared returns at lag >= 1
                                (GARCH: materially positive; GBM: ~0)
* ``autocorr``                — ACF of any series over multiple lags
* ``excess_kurtosis``         — fat tails (GBM: ~0; jumps: positive)
* ``regime_persistence``      — diag dominance of recovered transition modes
"""

from __future__ import annotations

import mlx.core as mx

EPS = 1e-12


def autocorr(x, max_lag: int = 1, axis: int = -1) -> mx.array:
    """Autocorrelation of ``x`` at lags 1..``max_lag``.

    Shape ``(..., max_lag)``. Fully vectorized over lags via shifting clock
    matrix ``idx + lag`` masked to valid entries.
    """
    max_lag = int(max_lag)
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    t = x.shape[axis]
    if t < 2:
        raise ValueError("need at least 2 observations on the time axis")

    xc = x - mx.mean(x, axis=axis, keepdims=True)
    denom = mx.sum(xc * xc, axis=axis) / t + EPS  # lag-0 autocovariance

    k = mx.arange(1, max_lag + 1, dtype=mx.int32)  # (max_lag,)
    i = mx.arange(t, dtype=mx.int32)
    idx = i[None, :] + k[:, None]                 # (max_lag, t) shifted clock
    valid = idx < t
    idx = mx.minimum(idx, t - 1)
    xshift = mx.take(xc, idx, axis=axis)          # (..., max_lag, t)
    xc_b = mx.expand_dims(xc, axis=-2)            # (..., 1, t)
    num = xc_b * xshift
    num = mx.where(valid, num, 0.0)
    cov = mx.sum(num, axis=axis) / (t - k)       # (..., max_lag)
    return cov / mx.expand_dims(denom, -1)


def vol_clustering_acf(r, lag: int = 1) -> mx.array:
    """ACF of squared (log-)returns at ``lag`` — the volatility-clustering
    diagnostic (e.g. ACF of squared residuals for GARCH families)."""
    return autocorr(r * r, max_lag=lag)[..., lag - 1]


def excess_kurtosis(x, axis: int = -1) -> mx.array:
    """Excess kurtosis = standardized 4th moment - 3. ~0 for normals."""
    mu = mx.mean(x, axis=axis, keepdims=True)
    sd = mx.sqrt(mx.mean((x - mu) ** 2, axis=axis, keepdims=True)) + EPS
    return mx.mean(((x - mu) / sd) ** 4, axis=axis) - 3.0


def skewness(x, axis: int = -1) -> mx.array:
    """Standardized 3rd moment. ~0 for symmetric distributions."""
    mu = mx.mean(x, axis=axis, keepdims=True)
    sd = mx.sqrt(mx.mean((x - mu) ** 2, axis=axis, keepdims=True)) + EPS
    return mx.mean(((x - mu) / sd) ** 3, axis=axis)


def regime_persistence(returns, n_segments: int, reference=None) -> mx.array:
    """Heuristic regime-persistence check: per-segment vol autocorrelation
    against segment-length noise.

    Splits ``returns`` into ``n_segments`` equal slices, computes the
    lag-1 ACF of squared returns per segment, and reports the *fraction of
    segments* whose ACF exceeds reference (default: 2 / sqrt(segment_len),
    the ~95% white-noise bound).

    Returns a single MLX scalar in [0, 1]: persistent-vol regimes push it
    well above 0, pure GBM keeps it near the chance rate.
    """
    n_segments = int(n_segments)
    if n_segments < 2:
        raise ValueError("n_segments must be >= 2")
    t = returns.shape[-1]
    seg_len = t // n_segments
    if seg_len < 8:
        raise ValueError("segments too short")
    sl = mx.arange(0, n_segments * seg_len, seg_len, dtype=mx.int32)
    grid = sl[None, :] + mx.arange(seg_len, dtype=mx.int32)[:, None]
    seg = mx.take(returns[..., : n_segments * seg_len], grid, axis=-1)
    acf = vol_clustering_acf(seg, lag=1)
    if reference is None:
        reference = 2.0 / (seg_len**0.5)
    return mx.mean(acf > reference)


def sharpe_year(returns, periods: int = 252) -> mx.array:
    """Annualized (population) Sharpe per path."""
    sd = mx.std(returns, axis=-1) + EPS
    return mx.mean(returns, axis=-1) / sd * (periods**0.5)