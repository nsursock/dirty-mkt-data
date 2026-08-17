"""P0 — OHLCV construction invariants (MLX, vectorized)."""

import mlx.core as mx

from dirty_mkt_data.api.generator import Generator
from dirty_mkt_data.core.gbm import GBM, TRADING_DAYS
from dirty_mkt_data.viz.ohlcv import build_ohlcv, from_dataset

SIG = 0.2
SEED = 9
T = 2048
B = 16


def _ohlcv(b=B, t=T):
    ds = Generator(GBM(mu=0.5 * SIG**2, sigma=SIG), seed=SEED).sample(t, n_paths=b)
    key = mx.random.key(SEED + 1)
    return build_ohlcv(ds.returns, sigma=SIG, dt=1.0 / TRADING_DAYS, s0=100.0, key=key), ds


def test_shapes():
    o, ds = _ohlcv()
    for a in (o.opens, o.highs, o.lows, o.closes, o.vols):
        assert a.shape == (B, T)


def test_high_low_sandwich_ohlc():
    o, _ = _ohlcv()
    assert bool(mx.all(o.highs >= mx.maximum(o.opens, o.closes) - 1e-6))
    assert bool(mx.all(o.lows <= mx.minimum(o.opens, o.closes) + 1e-6))
    assert bool(mx.all(o.highs >= o.lows))


def test_open_is_previous_close():
    o, _ = _ohlcv()
    assert bool(mx.all(o.opens[:, 1:] == o.closes[:, :-1]))


def test_lows_below_highs_strict_on_volatile_path():
    o, _ = _ohlcv()
    spread = o.highs - o.lows
    assert float(mx.mean(spread)) > 0
    assert bool(mx.all(mx.isfinite(o.highs)))


def test_volume_positive_and_moves_with_bar():
    o, _ = _ohlcv()
    assert bool(mx.all(o.vols > 0))
    small = mx.mean(o.vols[:, : T // 2])
    large = mx.mean(o.vols[:, T // 2:])
    assert float(small) > 0 and float(large) > 0


def test_from_dataset_closes_match_prices():
    ds = Generator(GBM(mu=0.5 * SIG**2, sigma=SIG), seed=SEED).sample(T, n_paths=4)
    o = from_dataset(ds, sigma=SIG, dt=1.0 / TRADING_DAYS, key=mx.random.key(SEED + 1))
    assert bool(mx.all(o.closes == ds.prices))


def test_deterministic_same_key():
    oa, _ = _ohlcv()
    ob, _ = _ohlcv()
    for fa, fb in ((oa.highs, ob.highs), (oa.vols, ob.vols), (oa.opens, ob.opens)):
        assert bool(mx.all(fa == fb))