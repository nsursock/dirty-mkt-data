"""Geometric Brownian Motion — the null-model control.

Log-returns are iid normal:

    r_t = (mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * eps_t,  eps_t ~ N(0, 1)

GBM intentionally has *no* volatility clustering, fat tails, or regime
persistence. Any strategy that finds edge in pure GBM output is broken
(see tests/test_gbm_control.py). Fully vectorized on MLX — no Python loops.
"""

from __future__ import annotations

import mlx.core as mx

from dirty_mkt_data.api.base import Dataset, Model
from dirty_mkt_data.eval.stylized_facts import excess_kurtosis, skewness

TRADING_DAYS = 252


class GBM(Model):
    namespace = "dirty_mkt_data.core.gbm.v1"

    def __init__(self, mu: float = 0.0, sigma: float = 0.2, s0: float = 100.0,
                 dt: float = 1.0 / TRADING_DAYS, **kwargs):
        super().__init__(mu=mu, sigma=sigma, s0=s0, dt=dt, **kwargs)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.s0 = float(s0)
        self.dt = float(dt)

    def _drift(self) -> float:
        return (self.mu - 0.5 * self.sigma**2) * self.dt

    def _scale(self) -> float:
        return self.sigma * (self.dt**0.5)

    def sample(self, n_steps: int, n_paths: int = 1, key=None) -> Dataset:
        n_steps = int(n_steps)
        n_paths = int(n_paths)
        if key is None:
            key = mx.random.key(0)
        eps = mx.random.normal((n_paths, n_steps), key=key)
        returns = self._drift() + self._scale() * eps
        prices = self.s0 * mx.exp(mx.cumsum(returns, axis=1))
        return Dataset(prices=prices, returns=returns,
                       n_paths=n_paths, n_steps=n_steps, seed=0)

    def fit(self, returns) -> "GBM":
        """MLE calibration from observed returns (time on last axis).

        Multiple paths are pooled into a single series before estimation.
        """
        r = mx.reshape(returns, (-1,))
        mean_r = mx.mean(r)
        var_r = mx.var(r)
        sigma2 = var_r / self.dt
        mu = mean_r / self.dt + 0.5 * sigma2
        return GBM(mu=float(mu), sigma=float(mx.sqrt(sigma2)), s0=self.s0, dt=self.dt)

    def log_likelihood(self, returns) -> mx.array:
        sd = self._scale()
        z = (returns - self._drift()) / sd
        per_step = -0.5 * (mx.log(2.0 * mx.pi) + 2.0 * mx.log(sd) + z * z)
        return mx.sum(per_step, axis=-1)

    def diagnostics(self, dataset: Dataset) -> dict[str, mx.array]:
        r = dataset.returns
        sd = mx.std(r, axis=-1)
        se = sd / (dataset.n_steps**0.5)
        t_mean = mx.mean(r, axis=-1) / se
        return {
            "mean_return": mx.mean(r, axis=-1),
            "std_return": sd,
            "skewness": skewness(r),
            "excess_kurtosis": excess_kurtosis(r),
            "mean_t_stat": t_mean,
            "seed": mx.array(dataset.seed, dtype=mx.int64),
        }