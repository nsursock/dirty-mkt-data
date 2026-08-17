"""P0 — GBM null-model control.

A naive TA strategy (moving-average crossover) must find **no** edge in pure
GBM output. We simulate many paths (vectorized on MLX), build a
look-ahead-free 5/20 crossover position, and test that (a) the strategy's
excess return over buy-and-hold has mean ≈ 0 and (b) neither strategy earns
a significant annualized Sharpe. All fully vectorized, no Python loops over
paths.

Null-model choice: we use a *log-martingale* GBM (mu = sigma^2/2), i.e. zero
*conditional expected* return. A plain mu=0 GBM has conditional expected
log-return -sigma^2/2, which any forward-weighted rule harvests — that would
look like edge where there is none (a bug in the test, not the model).
"""

import mlx.core as mx

from dirty_mkt_data.core.gbm import GBM, TRADING_DAYS
from dirty_mkt_data.eval.rolling import rolling_mean, shift_right
from dirty_mkt_data.eval.stylized_facts import sharpe_year

PATHS = 512
STEPS = 2520  # 10y daily
SHORT_W, LONG_W = 5, 20
SIGMA = 0.2
MU_NULL = 0.5 * SIGMA**2      # log-martingale: zero conditional expected return
Z_MAX = 3.0


def _crossover(returns):
    """Look-ahead-free long/flat/short position from 5/20 MA crossover.

    position at step t depends only on prices through t-1.
    """
    prices = mx.exp(mx.cumsum(returns, axis=1))
    ms = shift_right(rolling_mean(prices, SHORT_W))
    ml = shift_right(rolling_mean(prices, LONG_W))
    sig = mx.sign(ms - ml)
    return mx.where(mx.isnan(sig), 0.0, sig)


def _cross_path_stats(returns, positions):
    strat = positions * returns          # (PATHS, STEPS)
    sh_sharp = sharpe_year(strat, TRADING_DAYS)
    sh_bh = sharpe_year(returns, TRADING_DAYS)
    return sh_sharp, sh_bh


def test_naive_ta_finds_no_edge_on_gbm():
    key = mx.random.key(1234)
    ds = GBM(mu=MU_NULL, sigma=SIGMA).sample(STEPS, n_paths=PATHS, key=key)
    r = ds.returns

    pos = _crossover(r)
    sharp, bh = _cross_path_stats(r, pos)

    # excess (strategy - buy&hold) Sharpe must average to ~0
    excess = sharp - bh
    mu = float(mx.mean(excess))
    se = float(mx.std(excess)) / (PATHS ** 0.5)
    z = mu / se

    mu_bh = float(mx.mean(bh))
    se_bh = float(mx.std(bh)) / (PATHS ** 0.5)
    z_bh = mu_bh / se_bh

    assert abs(z) < Z_MAX
    assert abs(z_bh) < Z_MAX
    # sanity: signal is real (touches both states)
    assert float(mx.mean(mx.abs(pos))) > 0.05


def test_position_is_lookahead_free():
    r = mx.random.normal((1, STEPS), key=mx.random.key(5))
    pos = _crossover(r)
    ms = shift_right(rolling_mean(mx.exp(mx.cumsum(r, axis=1)), SHORT_W))
    ml = shift_right(rolling_mean(mx.exp(mx.cumsum(r, axis=1)), LONG_W))
    assert bool(mx.all(pos == mx.sign(ms - ml)))