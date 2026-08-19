"""P1 — ARGBM AR(1) log-returns: injectable momentum / mean-reversion.

Mirrors tests/test_gbm.py (reproducibility, shapes, moments) plus AR(1)
structure: phi=0 recovers GBM, |acf|=phi, variance preserved across phi,
log-likelihood peaks at the true phi, and fit roundtrips mu/sigma/phi.
"""

import mlx.core as mx

from dirty_mkt_data import Generator, Model
from dirty_mkt_data.api.seeding import SeedContract
from dirty_mkt_data.core.argbm import ARGBM, TRADING_DAYS
from dirty_mkt_data.eval.stylized_facts import autocorr

N = 1024
T = 1024
SEED = 11
SIG = 0.25


def _g(phi=0.0, mu=0.04, key=None):
    return ARGBM(mu=mu, sigma=SIG, phi=phi)

# --- interface ---------------------------------------------------------------


def test_is_model_with_versioned_namespace():
    m = ARGBM(phi=0.2)
    assert isinstance(m, Model)
    assert m.namespace == "dirty_mkt_data.core.argbm.v1"

# --- reproducibility ---------------------------------------------------------


def test_sample_reproducible_same_key():
    a = _g(phi=0.5).sample(T, n_paths=8, key=mx.random.key(SEED))
    b = _g(phi=0.5).sample(T, n_paths=8, key=mx.random.key(SEED))
    assert bool(mx.all(a.prices == b.prices))
    assert bool(mx.all(a.returns == b.returns))


def test_sample_different_key_differs():
    a = _g(phi=0.5).sample(T, n_paths=8, key=mx.random.key(SEED))
    b = _g(phi=0.5).sample(T, n_paths=8, key=mx.random.key(SEED + 1))
    assert not bool(mx.all(a.prices == b.prices))


def test_generator_reproducible_end_to_end():
    a = Generator(ARGBM(mu=0.02, sigma=0.15, phi=0.3), seed=42).sample(T, n_paths=8)
    b = Generator(ARGBM(mu=0.02, sigma=0.15, phi=0.3), seed=42).sample(T, n_paths=8)
    assert bool(mx.all(a.prices == b.prices))
    assert bool(mx.all(a.returns == b.returns))


def test_generator_different_seed_differs():
    a = Generator(ARGBM(phi=0.4), seed=1).sample(T, n_paths=8)
    b = Generator(ARGBM(phi=0.4), seed=2).sample(T, n_paths=8)
    assert not bool(mx.all(a.returns == b.returns))


def test_ar1_namespace_derives_distinct_deterministic_key():
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[1] / "src")
    code = (
        "from dirty_mkt_data.api.seeding import SeedContract; "
        f"print(SeedContract(42).digest('{ARGBM.namespace}'))"
    )
    out1 = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          check=True, env={**os.environ, "PYTHONPATH": src})
    out2 = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          check=True, env={**os.environ, "PYTHONPATH": src})
    assert out1.stdout == out2.stdout
    assert int(out1.stdout) == SeedContract(42).digest(ARGBM.namespace)
    k_ar = SeedContract(42).key(ARGBM.namespace)
    k_gbm = SeedContract(42).key("dirty_mkt_data.core.gbm.v1")
    assert bool(mx.any(k_ar != k_gbm))

# --- shapes & value sanity ---------------------------------------------------


def test_shapes():
    ds = _g(phi=0.3).sample(T, n_paths=8, key=mx.random.key(SEED))
    assert ds.prices.shape == (8, T)
    assert ds.returns.shape == (8, T)
    assert ds.n_paths == 8 and ds.n_steps == T


def test_prices_positive_and_endpoint_consistent():
    ds = _g(phi=0.3).sample(T, n_paths=8, key=mx.random.key(SEED))
    assert bool(mx.all(ds.prices > 0))
    log_growth = mx.sum(ds.returns, axis=1)
    expected = mx.log(ds.prices[:, -1] / 100.0)
    assert bool(mx.all(mx.abs(log_growth - expected) < 1e-3))


def test_rejects_nonstationary_phi():
    import pytest

    with pytest.raises(ValueError):
        ARGBM(phi=1.0)
    with pytest.raises(ValueError):
        ARGBM(phi=-1.0)

# --- phi = 0 recovers GBM ----------------------------------------------------


def test_zero_phi_matches_gbm_scale_and_whiteness():
    from dirty_mkt_data.core.gbm import GBM

    a = ARGBM(mu=0.04, sigma=SIG).sample(T, n_paths=N, key=mx.random.key(SEED))
    b = GBM(mu=0.04, sigma=SIG).sample(T, n_paths=N, key=mx.random.key(SEED))
    dt = 1.0 / TRADING_DAYS

    # variance-preserving: same per-step scale as GBM
    assert abs(float(mx.std(a.returns)) - float(mx.std(b.returns))) < 1e-4
    assert abs(float(mx.std(a.returns)) - SIG * dt**0.5) < 0.002

    # phi=0 reduces to exact GBM given the same key
    assert bool(mx.all(a.returns == b.returns))

    # iid: zero lag-1 autocorrelation and no exploitable structure
    lag1 = autocorr(mx.reshape(a.returns, (-1,)), max_lag=1)[0]
    assert abs(float(lag1)) < 0.01

# --- AR(1) structure ---------------------------------------------------------


def test_positive_phi_gives_momentum():
    ds = _g(phi=0.5).sample(T, n_paths=N, key=mx.random.key(SEED))
    lag1 = autocorr(mx.reshape(ds.returns, (-1,)), max_lag=1)[0]
    assert 0.35 < float(lag1) < 0.65


def test_negative_phi_gives_mean_reversion():
    ds = _g(phi=-0.5).sample(T, n_paths=N, key=mx.random.key(SEED))
    lag1 = autocorr(mx.reshape(ds.returns, (-1,)), max_lag=1)[0]
    assert -0.65 < float(lag1) < -0.35
    assert float(lag1) < 0


def test_per_step_std_variance_preserved_across_phi():
    sts = []
    for phi in (0.0, +0.5, -0.5, +0.8):
        ds = _g(phi=phi).sample(T, n_paths=N, key=mx.random.key(SEED))
        sts.append(float(mx.std(ds.returns)))
    assert max(sts) - min(sts) < 0.002

# --- fit & likelihood --------------------------------------------------------


def test_fit_recovers_parameters():
    # drift is tiny vs per-step vol: need ~2M samples for a tight mu estimate
    ds = _g(mu=0.04, phi=0.4).sample(8192, n_paths=32, key=mx.random.key(SEED))
    fitted = _g(mu=0.04, phi=0.4).fit(ds.returns)
    assert abs(fitted.sigma - SIG) < 1e-2
    assert abs(fitted.mu - 0.04) < 2e-2
    assert abs(fitted.phi - 0.4) < 2e-2


def test_log_likelihood_peaks_at_true_phi():
    ds = _g(mu=0.04, phi=0.4).sample(T, n_paths=8, key=mx.random.key(SEED))
    ll_true = _g(mu=0.04, phi=0.4).log_likelihood(ds.returns)
    ll_wrong = _g(mu=0.04, phi=-0.4).log_likelihood(ds.returns)
    ll_null = _g(mu=0.04, phi=0.0).log_likelihood(ds.returns)
    assert bool(mx.all(mx.isfinite(ll_true)))
    assert float(mx.mean(ll_true)) > float(mx.mean(ll_wrong))
    assert float(mx.mean(ll_true)) > float(mx.mean(ll_null))


def test_diagnostics_has_ar1_and_matches_lag1_acf():
    ds = Generator(ARGBM(mu=0.03, sigma=0.2, phi=0.5), seed=7).sample(T, n_paths=16)
    diag = ARGBM(mu=0.03, sigma=0.2, phi=0.5).diagnostics(ds)
    assert "ar1" in diag
    lag1 = autocorr(ds.returns, max_lag=1)[..., 0]
    assert bool(mx.all(mx.abs(diag["ar1"] - lag1) < 1e-6))
    for v in diag.values():
        assert isinstance(v, mx.array)