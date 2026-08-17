"""P0 — common Model interface used by every generator."""

import mlx.core as mx

from dirty_mkt_data import Generator, Model
from dirty_mkt_data.core.gbm import GBM

T = 1024


def test_model_is_registered():
    m = GBM()
    assert isinstance(m, Model)
    assert hasattr(m, "namespace") and m.namespace.startswith("dirty_mkt_data")


def test_interface_methods_present_and_callable():
    m = GBM()
    for name in ("sample", "fit", "log_likelihood", "diagnostics"):
        assert callable(getattr(m, name, None))


def test_generator_composes_model_and_records_seed():
    g = Generator(GBM(), seed=99)
    ds = g.sample(T, n_paths=4)
    assert ds.seed == 99
    assert ds.prices.shape == (4, T)
    assert ds.returns.shape == (4, T)


def test_generator_reproducible_end_to_end():
    a = Generator(GBM(mu=0.02, sigma=0.15), seed=42).sample(T, n_paths=8)
    b = Generator(GBM(mu=0.02, sigma=0.15), seed=42).sample(T, n_paths=8)
    assert bool(mx.all(a.prices == b.prices))
    assert bool(mx.all(a.returns == b.returns))


def test_generator_different_seed_differs():
    a = Generator(GBM(), seed=1).sample(T, n_paths=8)
    b = Generator(GBM(), seed=2).sample(T, n_paths=8)
    assert not bool(mx.all(a.returns == b.returns))


def test_generator_run_id_gives_independent_streams():
    g = Generator(GBM(), seed=42)
    a = g.sample(T, n_paths=8, run_id=0)
    b = g.sample(T, n_paths=8, run_id=1)
    assert not bool(mx.all(a.returns == b.returns))
    # same run_id stable across calls
    c = g.sample(T, n_paths=8, run_id=0)
    assert bool(mx.all(a.returns == c.returns))


def test_generator_rejects_non_model():
    import pytest

    with pytest.raises(TypeError):
        Generator(object(), seed=1)


def test_diagnostics_returns_mlx_dict():
    ds = Generator(GBM(mu=0.03, sigma=0.2), seed=7).sample(T, n_paths=16)
    diag = GBM(mu=0.03, sigma=0.2).diagnostics(ds)
    assert set(diag) >= {"mean_return", "std_return", "skewness", "excess_kurtosis"}
    for v in diag.values():
        assert isinstance(v, mx.array)