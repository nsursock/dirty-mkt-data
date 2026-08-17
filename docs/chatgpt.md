That makes a lot of sense. **`dirty-mkt-data`** is actually a cleaner home for the synthetic-market side of what you've been building.

I'd make it a standalone **market-data generation / stochastic-process research project**, rather than tying it to the trading bot.

### Core scope

Start with the classical families:

* **GBM** — Geometric Brownian Motion
* **GBM variants** — drift/volatility regimes, time-varying parameters
* **GARCH** — volatility clustering
* **EGARCH / GJR-GARCH** — leverage/asymmetry effects
* **Heston** — stochastic volatility
* **Merton jump-diffusion** — discontinuous jumps
* **Kou jump-diffusion** — asymmetric double-exponential jumps
* **Bates** — Heston + jumps
* **OU / CIR** — mean-reverting processes
* **Regime-switching models** — e.g. Markov switching
* **Hawkes processes** — clustered/self-exciting events
* **Multivariate models** — correlated assets, dynamic correlations
* **Copula-based generators** — cross-asset dependence/tail dependence

Then eventually the interesting part: **compositions**.

For example:

`regime → stochastic volatility → jumps → cross-asset correlation → microstructure/noise`

That gives you synthetic tapes that can progressively move from *toy stochastic process* → *realistic market simulator*.

### The really valuable part

I wouldn't make it merely a collection of generators.

Give the project a common interface, something like:

```text
Model
 ├── fit(real_data)
 ├── sample(n_steps, seed)
 ├── log_likelihood(...)
 └── diagnostics(...)
```

and a common evaluation framework:

```text
Synthetic
    ↓
Distributional tests
    ↓
Stylized facts
    ↓
Temporal dependence
    ↓
Volatility behavior
    ↓
Tail behavior
    ↓
Cross-asset dependence
    ↓
Downstream ML usefulness
```

So you can eventually answer much more interesting questions than *"does this look like BTC?"*

For example:

> **Does a SAC/PPO agent trained on synthetic data generalize to unseen real regimes?**

That's where `dirty-mkt-data` becomes genuinely useful to your trading project.

And given your current HRL work, I'd keep the dependency direction:

```text
dirty-mkt-data
       ↓
iso-trading-bot
       ↓
RL / strategy experiments
```

rather than putting the generators inside the bot. It gives you a reusable **synthetic financial-data laboratory** that you can use for RL, supervised learning, stress testing, backtesting, and eventually benchmarking different algorithms.
