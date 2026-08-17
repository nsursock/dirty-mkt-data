Good, that simplifies things a lot — pure focus on generation quality and reusability, no backtest/eval baggage attached. Here's how I'd structure it.

**Core design principle:** layered composability. Each layer is a pure function/class that takes a price/return series and regime metadata and returns a modified one, so you can turn dirty-ness on/off independently and know exactly what's contaminating your results later.

```
dirty-mkt-data/
  core/
    gbm.py          # baseline null-model generator (log-normal returns, no clustering)
    garch.py        # GARCH(1,1)/EGARCH/GJR variants, vol clustering, fat tails
    regimes.py       # regime-switching layer (Markov chain over N regimes, your 8-regime carryover)
    jumps.py         # Poisson/Hawkes jump-diffusion overlay, optional
  contamination/      # "dirty" layer — data quality defects, separate from price dynamics
    gaps.py           # missing bars, exchange outages
    noise.py          # microstructure noise, bad ticks, duplicate timestamps
    clock.py          # timestamp skew/misalignment
  api/
    generator.py      # top-level Generator class, composes core + contamination
    seeding.py         # deterministic seed -> dataset contract
  tests/
    test_statistical_properties.py  # validate stylized facts per layer
```

**Key architectural decisions I'd lock in now:**

1. **Seed contract first.** `Generator(seed=42, regime_sequence=..., contamination=None)` should be byte-for-byte reproducible forever. Write this test before writing any regime logic — it's the thing that's expensive to retrofit.

2. **GBM stays as the null-model control**, not a "regime" — it's your falsification baseline (a strategy that finds edge in pure GBM output is broken, full stop). Keep it dead simple and validate that a naive TA feature set truly finds nothing there.

3. **GARCH + regime-switching is the real workhorse.** Port your 8-regime GARCH generator from `iso-trading-bot` here almost as-is (it's already validated), but strip any iso-trading-bot-specific coupling (config paths, MLX-specific assumptions) so it's a clean standalone dependency.

4. **Contamination is opt-in and stacked last**, applied to whatever clean series core/ produced — never baked into the regime math. This way "how much does exchange-outage noise degrade my strategy's Sharpe" is a controlled experiment, not a confound.

5. **Validation suite from day one** — for each generator, assert the stylized facts it's supposed to produce (vol clustering via ACF of squared returns, fat tails via excess kurtosis, regime persistence via transition matrix recovery) so you're not trusting "looks right" by eye.

Want me to start scaffolding this (repo structure + the GBM and GARCH core modules), or do you want to first pull over and clean up the existing 8-regime GARCH code from iso-trading-bot as the seed for `core/garch.py` and `core/regimes.py`?