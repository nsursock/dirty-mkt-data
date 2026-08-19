# dirty-mkt-data

Synthetic financial market-data generator. **Apple MLX only** — no NumPy, no
Pandas, no Python loops in the numerical kernels. Fully vectorized on the
Apple Neural Engine / GPU, with a strict reproducibility contract.

## Why

Real market data is dirty: missing bars, microstructure noise, regime shifts,
fat tails, volatility clustering. `dirty-mkt-data` generates synthetic price
series you control precisely — clean or deliberately contaminated — so you can
run controlled experiments instead of guessing:

> *"How much does exchange-outage noise degrade my strategy's Sharpe?"*

The goal is a reusable **synthetic financial-data laboratory** for RL, stress
testing, backtesting, and benchmarking — not a black-box simulator.

## Status

All **P0** items and the **P1 AR(1) alpha-injection** model are implemented
and tested (57 tests green). See [Roadmap](#roadmap).

## Setup

```bash
python -m venv venv
venv/bin/pip install -e ".[dev]"     # mlx, pyyaml, pytest, plotly, kaleido
```

> Requires Python 3.10+ and Apple Silicon. Static images (PNG) additionally
> need Chrome-for-Testing for kaleido:
> `venv/bin/python -m pip install kaleido && plotly_get_chrome`

## Quick start

```python
from dirty_mkt_data import Generator
from dirty_mkt_data.core.gbm import GBM

# Reproducible synthetic GBM sample: same seed => byte-identical output
ds = Generator(GBM(mu=0.5 * 0.2**2, sigma=0.2), seed=42).sample(2520, n_paths=8)
print(ds.prices.shape)   # (8, 2520)

# Calibrate / evaluate
fitted = GBM().fit(ds.returns)
ll = GBM(mu=0.0, sigma=0.2).log_likelihood(ds.returns)
diag = GBM(sigma=0.2).diagnostics(ds)   # mean/std/skewness/kurtosis/t-stat
```

### Injectable alpha: AR(1) log-returns

`ARGBM` is a variance-preserving AR(1) wrapper on GBM: same per-step return
variance as GBM for every `phi`, only the temporal structure changes.

```python
from dirty_mkt_data.core.argbm import ARGBM

# momentum (phi > 0) or mean-reversion (phi < 0); phi = 0 == GBM exactly
ds = Generator(ARGBM(mu=0.0, sigma=0.2, phi=0.5), seed=42).sample(2520, n_paths=8)
diag = ARGBM(sigma=0.2, phi=0.5).diagnostics(ds)   # ... + lag-1 "ar1" ~ phi

# calibrate (Yule-Walker) and evaluate the AR(1) model against a series
fitted = ARGBM(phi=0.5).fit(ds.returns)
ll = ARGBM(phi=0.5).log_likelihood(ds.returns)     # conditional AR(1) + stationary init
```

### Render OHLCV charts

```bash
# All parameters live in configs/inspect.yaml
venv/bin/python scripts/inspect_gbm.py
```

Generates a candlestick + volume figure per theme (`ghibli`, `synthwave`,
`valorant`) in JetBrains Mono into `figures/`:

```bash
venv/bin/python scripts/inspect_gbm.py --steps 500 --out figures/500c --html
```

## Architecture

Layered composability: pure-function layers over the time axis, so dirtiness
can be toggled on/off and you always know what is contaminating your results.

```
src/dirty_mkt_data/
  api/
    base.py         # Model / Contamination interfaces, Dataset
    seeding.py      # SeedContract: (seed, namespace) -> deterministic MLX key
    generator.py    # Generator: composes model (+ optional contamination)
  core/
    gbm.py          # GBM — null-model control (done, P0)
    argbm.py        # ARGBM — variance-preserving AR(1) alpha injection (done, P1)
    garch.py        # GARCH(1,1)/EGARCH/GJR — P1
    regimes.py      # Markov regime-switching (8-regime port) — P1
    jumps.py        # Poisson/Hawkes jump-diffusion — P2
  contamination/    # opt-in defects applied LAST — P1
    gaps.py         # missing bars, exchange outages
    noise.py        # microstructure noise, bad ticks
    clock.py        # timestamp skew / misalignment
  eval/
    stylized_facts.py  # ACF, vol clustering, kurtosis, regime persistence
    rolling.py         # vectorized rolling windows (cumsum-based)
  viz/                 # OHLCV construction + plotly themes (P0)
tests/                 # pytest suite (41 tests)
configs/               # YAML configs
scripts/               # CLI helpers
```

Every generator implements the common `Model` interface — `fit(real_data)`,
`sample(n_steps, n_paths, key)`, `log_likelihood(...)`, `diagnostics(...)` —
so models are swappable and composable through one pipeline.

## Reproducibility contract

`Generator(model, seed)` is **byte-for-byte reproducible forever**: same
(seed, namespace) always derives the same MLX random key. Versioned
namespaces (e.g. `dirty_mkt_data.core.gbm.v1`) mean you bump the namespace
whenever a sampling algorithm changes — like a dataset schema version.
Verified by an in-process test and a cross-process (subprocess) test.

## Roadmap

| Priority | Item | Status |
|----------|------|--------|
| **P0** | Repo scaffold (`core/ contamination/ api/ tests/`) | ✅ done |
| **P0** | Seed/reproducibility contract | ✅ done |
| **P0** | GBM generator (null-model control) | ✅ done |
| **P0** | GBM-as-control: naive TA finds no edge on GBM | ✅ done |
| **P0** | Common `Model` interface (fit/sample/log_likelihood/diagnostics) | ✅ done |
| **P0** | Stylized-facts validation framework | ✅ done |
| **P1** | AR(1) log-returns (ARGBM, variance-preserving injectable alpha) | ✅ done |
| **P1** | GARCH(1,1) with volatility clustering | 🔜 next |
| **P1** | Markov regime-switching (8-regime, port from iso-trading-bot) | ⬜ |
| **P1** | GARCH + regime composition | ⬜ |
| **P1** | Contamination layer: gaps, noise, clock (opt-in, applied last) | ⬜ |
| **P1** | Standalone packaging / dependency direction | ⬜ |
| **P2** | GARCH asymmetry (EGARCH, GJR-GARCH, APARCH) | ⬜ |
| **P2** | Jump diffusion (Merton, Kou, Bates) | ⬜ |
| **P2** | Stochastic volatility (Heston) + mean-reverting (OU/CIR) | ⬜ |
| **P2** | Composition pipeline `regime → SV → jumps → cross-asset → noise` | ⬜ |
| **P2** | Dirty-data extras: non-synchronous sampling, imputation | ⬜ |
| **P3** | Generative-AI models (TimeGAN, MacroVAE, diffusion) | ⬜ |
| **P3** | Standardized benchmarks (FinStressTS, CTBench) | ⬜ |
| **P3** | Multi-asset: DCC-MGARCH, copulas | ⬜ |
| **P3** | Downstream ML/RL validation (TCN/LSTM/Transformer, agents) | ⬜ |

The null-model control is a design cornerstone: any strategy that earns an
edge on pure GBM output is broken, full stop. Real dynamics (ARGBM now,
GARCH + regimes later) should legitimately break that null — that is the
falsification making the benchmark meaningful.

## Tests

```bash
venv/bin/python -m pytest        # 57 tests: seed contract, GBM stats/control,
                                 # ARGBM AR(1), model interface, stylized facts, OHLCV
```

## Layout

| Path | Role |
|------|------|
| `src/dirty_mkt_data/` | Library code (MLX only) |
| `tests/` | Pytest suite |
| `configs/` | YAML configs |
| `scripts/` | CLI helpers |
| `docs/` | Notes, priority lists |
| `figures/` | Example OHLCV renders |
