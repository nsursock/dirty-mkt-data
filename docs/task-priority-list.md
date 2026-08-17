# dirty-mkt-data — Task Priority List

Build-order task list distilled from LLM design reviews in `docs/` (Claude, Gemini, DeepSeek, ChatGPT). Tasks are ordered by priority, then by dependency (earlier tasks unblock later ones). **Validation** describes the test/check that proves the task is done.

## Task List

| Priority | Task | Validation | Impact | Source(s) |
|----------|------|------------|--------|-----------|
| **P0** | Scaffold repo: `core/`, `contamination/`, `api/`, `tests/` | Clean package layout, imports resolve, empty test suites run | Gives every later task a designated home; keeps layers separable from day one | Claude |
| **P0** | Define seed contract + reproducibility test | `Generator(seed=42, ...)` produces byte-identical output across separate runs and versions | Guards determinism before any generator logic exists — cheapest time to lock it in | Claude |
| **P0** | Implement GBM generator (constant vol, log-normal returns) | Unit tests: sample shape, mean/vol recovered for large `n`; good first test target | Working null-model baseline; simplest piece, ideal to learn the test harness on | Claude, DeepSeek, ChatGPT |
| **P0** | GBM-as-control validation | Assert a naive TA feature set (e.g., moving-average crossovers) finds **no** edge on pure GBM output | Any strategy that profits on GBM is broken — catches false signal early | Claude |
| **P0** | Common `Model` interface | Every generator implements `fit(real_data)`, `sample(n_steps, seed)`, `log_likelihood(...)`, `diagnostics(...)`; tests exercise interface on all models | One evaluation pipeline serves all models; models are swappable/composable | ChatGPT |
| **P0** | Stylized-facts validation framework | Ready-made tests for vol clustering (ACF of squared returns), fat tails (excess kurtosis), leverage effect | Standardized quality gate changeable per generator; no more "looks right" eyeballing | Claude, ChatGPT |
| **P1** | Implement GARCH(1,1) with vol clustering | Passes vol-clustering check (significant ACF on squared returns) and fat-tail check (positive excess kurtosis); `fit` recovers params on synthetic series | Core workhorse for realistic volatility | Claude, Gemini, DeepSeek, ChatGPT |
| **P1** | Implement Markov regime-switching (8-regime) — port from iso-trading-bot | Transition-matrix recovery test; regime labels match known ground-truth sequence; strip iso-trading-bot coupling (config paths, MLX) | Reuses already-validated code; backbone of the "real workhorse" | Claude |
| **P1** | GARCH + regime composition | Series from the composed generator passes vol-clustering, fat-tail, AND regime-persistence checks simultaneously | Both dynamics coexist cleanly; template for future compositions | Claude |
| **P1** | Contamination layer (gaps, noise, clock) — opt-in, applied last | Each defect toggles independently on the same clean seed; `Contamination=None` reproduces clean output exactly | Controlled degradation experiments ("outage noise vs. Sharpe") without confounding | Claude, Gemini |
| **P1** | Standalone packaging / dependency direction | Package imports with no iso-trading-bot dependency; `dirty-mkt-data` → `iso-trading-bot` dependency enforced | Reusable synthetic-data lab usable by RL and backtesting without coupling | ChatGPT |
| **P2** | GARCH asymmetry extensions (EGARCH, GJR-GARCH, optional APARCH) | Leverage-effect test: negative shocks raise vol more than positive; baseline vs. asymmetric models compared | More realistic vol response to bad/good news | Gemini, DeepSeek, ChatGPT |
| **P2** | Jump-diffusion (Merton, Kou, Bates) | Fat-tail/kurtosis check materially exceeds pure GBM; jump frequency matches Poisson setup; individual jumps visible in samples | Sudden moves and crash scenarios for stress testing | Gemini, DeepSeek, ChatGPT |
| **P2** | Stochastic-vol (Heston) + mean-reverting (OU/CIR) | SV paths show non-autocorrelated vol (latent vol); OU/CIR paths revert to long-run mean (verify via sample stats) | Covers latent-volatility and mean-reversion stylized facts | Gemini, DeepSeek, ChatGPT |
| **P2** | Composition pipeline | `regime → SV → jumps → cross-asset → microstructure/noise` runs end-to-end; each stage's new artifact detectable | Progressive realism from toy process to market-simulator-grade tape | ChatGPT |
| **P2** | Dirty-data extras: non-synchronous/irregular sampling + imputation | Forward-fill/tick-time resampling produces aligned series; imputed gaps preserve adjacent distribution stats | Real-world data handling layer complete | Gemini |
| **P3** | Generative-AI models (TimeGAN / MacroVAE / diffusion) | Generated series pass the same stylized-facts suite; distributional tests vs. classic baselines | SOTA realism; synthetic training data for agents | Gemini, DeepSeek |
| **P3** | Standardized benchmarks (FinStressTS, CTBench) | Models scored on the 30 FinStressTS environments and CTBench trading/risk metrics | Systematic cross-model comparison rather than ad-hoc eval | DeepSeek |
| **P3** | Multi-asset: DCC-MGARCH + copulas | Correlated samples reproduce target correlation matrix; copula-based tail dependence (e.g., upper/lower tail) matches config | Portfolio-level and cross-asset risk scenarios | Gemini, ChatGPT |
| **P3** | Downstream ML/RL validation (TCN/LSTM/Transformer, JDAPP) | Agent (SAC/PPO) trained on synthetic data generalizes to unseen real regimes; forecasters beat GARCH baseline on MSE/VaR | Proves project value for the trading bot, not just data aesthetics | ChatGPT, DeepSeek |

## Suggested Build Order

1. **P0 tasks in listed order** (scaffold → seed test → GBM → GBM control → interface → eval framework). GBM is intentionally the first real generator: simple, unit-test friendly, and every later model is validated against it.
2. **P1** ports the validated GARCH/regime core and adds contamination + packaging — the deliverable becomes independently usable.
3. **P2–P3** are optional expansion tracks; pull them in as research goals warrant.