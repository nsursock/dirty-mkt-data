"""Vectorized rolling-window helpers on MLX.

Used by the GBM-control validation and the stylized-facts framework.
All kernels operate on the *last* axis (time) of an array of arbitrary rank,
e.g. ``(n_paths, n_steps)``. No Python loops; windows via cumulative sums
and ``mx.take``.
"""

from __future__ import annotations

import mlx.core as mx


def rolling_mean(x, window: int) -> mx.array:
    """Trailing ``window``-step mean, same shape/length as ``x``.

    Entry ``t`` = mean of ``x[t-window+1 : t+1]`` (partial-window means at
    the head). Vectorized with cumulative sums.
    """
    window = int(window)
    if window < 1:
        raise ValueError("window must be >= 1")
    if x.shape[-1] < 1:
        raise ValueError("x must have at least one step on the last axis")

    cs = mx.cumsum(x, axis=-1)
    lead = mx.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    cs = mx.concatenate([lead, cs], axis=-1)  # cs[t] = sum_{j<t} x_j
    t = x.shape[-1]
    idx1 = mx.arange(1, t + 1, dtype=mx.int32)
    idx0 = mx.maximum(idx1 - window, 0)
    num = mx.take(cs, idx1, axis=-1) - mx.take(cs, idx0, axis=-1)
    den = mx.minimum(idx1, window)
    return num / den


def shift_right(x, fill: float | int = 0.0) -> mx.array:
    """Shift along the last axis by one, carrying the first element forward.

    Used to apply a signal computed through ``t-1`` at step ``t`` without
    look-ahead.
    """
    carry = x[..., :1]
    return mx.concatenate([carry, x[..., :-1]], axis=-1)