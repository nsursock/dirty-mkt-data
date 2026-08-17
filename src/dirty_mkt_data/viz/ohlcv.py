"""Vectorized OHLCV bars from GBM / any return series (MLX only).

Bar ``t`` spans the interval ``(t-1, t]``:

    open[t]   = close[t-1]                      (t>=1)
    open[0]   = first close (no prior bar)
    close[t]  = price[t]
    high[t]   = max(open, close) * exp(+|eps_h| * k)
    low[t]    = min(open, close) * exp(-|eps_l| * k)
    volume[t] = base * (1 + 0.9 * move_score) * exp(sigma_v * eps_v)

``k = exc_scaling * sigma * sqrt(dt)`` controls the within-bar excursion. All
stochastic inputs come from independent MLX keys; no Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from dirty_mkt_data.api.base import Dataset


@dataclass(frozen=True)
class OHLCV:
    """OHLCV bars over time axis. ``opens``/``closes``/``highs``/``lows``/``vols``
    are ``(n_paths, n_steps)`` MLX arrays."""

    opens: mx.array
    highs: mx.array
    lows: mx.array
    closes: mx.array
    vols: mx.array


def build_ohlcv(
    returns: mx.array,
    prices: mx.array | None = None,
    s0: float = 100.0,
    sigma: float = 0.2,
    dt: float = 1.0 / 252.0,
    base_volume: float = 1_000_000.0,
    exc_scaling: float = 1.2,
    vol_noise: float = 0.15,
    key=None,
) -> OHLCV:
    """Build OHLCV bars from log-returns ``(n_paths, n_steps)``.

    If ``prices`` is given it becomes the close series (opens/lows/highs are
    derived from it); otherwise constructed from ``returns``.
    """
    n_paths, n_steps = returns.shape
    if key is None:
        key = mx.random.key(0)
    keys = mx.random.split(key, 3)
    k_h = exc_scaling * sigma * (dt**0.5)
    bar_vol = sigma * (dt**0.5)

    if prices is None:
        prices = s0 * mx.exp(mx.cumsum(returns, axis=1))
    closes = prices
    opens = mx.concatenate([closes[:, :1], closes[:, :-1]], axis=1)

    eps_h = mx.abs(mx.random.normal((n_paths, n_steps), key=keys[0]))
    eps_l = mx.abs(mx.random.normal((n_paths, n_steps), key=keys[1]))
    highs = mx.maximum(opens, closes) * mx.exp(eps_h * k_h)
    lows = mx.minimum(opens, closes) * mx.exp(-eps_l * k_h)

    move_score = mx.abs(returns) / (bar_vol + 1e-12)
    eps_v = mx.random.normal((n_paths, n_steps), key=keys[2])
    vols = base_volume * (1.0 + 0.9 * move_score) * mx.exp(vol_noise * eps_v)
    vols = mx.maximum(vols, 0.2 * base_volume)

    return OHLCV(opens=opens, highs=highs, lows=lows, closes=closes, vols=vols)


def from_dataset(dataset: Dataset, **kwargs) -> OHLCV:
    """OHLCV whose closes are the Dataset's own prices (s0 comes from the
    first price of the first path)."""
    s0 = float(dataset.prices[0, 0])
    return build_ohlcv(dataset.returns, prices=dataset.prices, s0=s0, **kwargs)