"""P0 — GBM generator correctness (MLX, fully vectorized)."""

import mlx.core as mx

from dirty_mkt_data.core.gbm import GBM, TRADING_DAYS

N = 4096
T = 4096
SEED = 11
SIG = 0.25


def _g(key=None):
    return GBM(mu=0.04, sigma=SIG, s0=100.0)

# --- reproducibility ---------------------------------------------------------


def test_sample_reproducible_same_key():
    a = _g().sample(T, n_paths=8, key=mx.random.key(SEED))
    b = _g().sample(T, n_paths=8, key=mx.random.key(SEED))
    assert bool(mx.all(a.prices == b.prices))
    assert bool(mx.all(a.returns == b.returns))


def test_sample_different_key_differs():
    a = _g().sample(T, n_paths=8, key=mx.random.key(SEED))
    b = _g().sample(T, n_paths=8, key=mx.random.key(SEED + 1))
    assert not bool(mx.all(a.prices == b.prices))

# --- shapes & value sanity ---------------------------------------------------


def test_shapes():
    ds = _g().sample(T, n_paths=8, key=mx.random.key(SEED))
    assert ds.prices.shape == (8, T)
    assert ds.returns.shape == (8, T)
    assert ds.n_paths == 8 and ds.n_steps == T


def test_prices_positive_and_endpoint_consistent():
    ds = _g().sample(T, n_paths=8, key=mx.random.key(SEED))
    assert bool(mx.all(ds.prices > 0))
    log_growth = mx.sum(ds.returns, axis=1)
    expected = mx.log(ds.prices[:, -1] / 100.0)
    assert bool(mx.all(mx.abs(log_growth - expected) < 1e-3))

# --- statistical properties of log-returns -----------------------------------


def test_log_returns_near_gaussian_moments():
    ds = _g().sample(T, n_paths=N, key=mx.random.key(SEED))
    r = ds.returns
    dt = 1.0 / TRADING_DAYS

    mu_hat = float(mx.mean(r))
    sig_hat = float(mx.std(r))
    assert abs(mu_hat - (0.04 - 0.5 * SIG**2) * dt) < 0.0003
    assert abs(sig_hat - SIG * dt**0.5) < 0.002

    kurt = float(mx.mean((r - mx.mean(r)) ** 4) / (mx.var(r) ** 2)) - 3.0
    assert abs(kurt) < 0.1

    skew = float(mx.mean((r - mx.mean(r)) ** 3) / (mx.var(r) ** 1.5))
    assert abs(skew) < 0.05


def test_no_autocorrelation_in_returns():
    ds = _g().sample(T, n_paths=N, key=mx.random.key(SEED))
    r_flat = mx.reshape(ds.returns, (-1,))
    lag1 = float(mx.mean(r_flat[1:] * r_flat[:-1]) / (mx.var(r_flat)))
    assert abs(lag1) < 0.01


def test_fit_recovers_parameters():
    ds = _g().sample(T, n_paths=8, key=mx.random.key(SEED))
    fitted = _g().fit(ds.returns)
    assert abs(fitted.sigma - SIG) < 1e-2
    assert abs(fitted.mu - 0.04) < 1e-2


def test_log_likelihood_finite_and_mle_superior():
    ds = _g().sample(T, n_paths=8, key=mx.random.key(SEED))
    true_ll = _g().log_likelihood(ds.returns)
    wrong_ll = GBM(mu=0.10, sigma=SIG).log_likelihood(ds.returns)
    assert bool(mx.all(mx.isfinite(true_ll)))
    assert float(mx.mean(true_ll)) > float(mx.mean(wrong_ll))