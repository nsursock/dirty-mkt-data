"""AR(1) autoregressive log-returns with injectable alpha.

    r_t = phi * r_{t-1} + (mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * sqrt(1 - phi^2) * eps_t,
          eps_t ~ N(0, 1)

The ``sqrt(1 - phi^2)`` factor keeps ``Var(r_t) = sigma^2 * dt`` for every
``phi``, so cross-phi comparisons are apples-to-apples (only temporal
structure changes): ``phi = 0`` recovers GBM exactly (iid), ``phi > 0``
momentum (persistent trends), ``phi < 0`` mean-reversion.

The time recursion is a short Python loop vectorized over the path axis
(eps is drawn up front in one shot), consistent with the EMA-style kernels.
"""

from __future__ import annotations

import mlx.core as mx

from dirty_mkt_data.api.base import Dataset, Model
from dirty_mkt_data.eval.stylized_facts import autocorr, excess_kurtosis, skewness

TRADING_DAYS = 252


class ARGBM(Model):
    namespace = "dirty_mkt_data.core.argbm.v1"

    def __init__(self, mu: float = 0.0, sigma: float = 0.2, s0: float = 100.0,
                 phi: float = 0.0, dt: float = 1.0 / TRADING_DAYS, **kwargs):
        super().__init__(mu=mu, sigma=sigma, s0=s0, phi=phi, dt=dt, **kwargs)
        if not (-1.0 < float(phi) < 1.0):
            raise ValueError("phi must lie in (-1, 1) for stationarity")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.s0 = float(s0)
        self.phi = float(phi)
        self.dt = float(dt)

    def _drift(self) -> float:
        return (self.mu - 0.5 * self.sigma**2) * self.dt

    def _scale(self) -> float:
        return self.sigma * (self.dt**0.5) * ((1.0 - self.phi**2) ** 0.5)

    def _stat_mean(self) -> float:
        return self._drift() / (1.0 - self.phi)

    def sample(self, n_steps: int, n_paths: int = 1, key=None) -> Dataset:
        n_steps = int(n_steps)
        n_paths = int(n_paths)
        if key is None:
            key = mx.random.key(0)
        eps = mx.random.normal((n_paths, n_steps), key=key)
        c = self._drift()
        sc = self._scale()
        base = self.sigma * (self.dt**0.5)
        cols = [self._stat_mean() + base * eps[:, 0]]
        for t in range(1, n_steps):
            prev = self.phi * cols[-1] + c + sc * eps[:, t]
            cols.append(prev)
        returns = mx.stack(cols, axis=1)
        prices = self.s0 * mx.exp(mx.cumsum(returns, axis=1))
        return Dataset(prices=prices, returns=returns,
                       n_paths=n_paths, n_steps=n_steps, seed=0)

    def fit(self, returns) -> "ARGBM":
        """Yule-Walker MLE calibration from observed returns (time last axis).

        Multiple paths are pooled into a single series before estimation.
        """
        r = mx.reshape(returns, (-1,))
        mean_r = mx.mean(r)
        var_r = mx.var(r)
        r_cent = r - mean_r
        denom = mx.sum(r_cent[:-1] ** 2) + 1e-12
        phi = mx.clip(mx.sum(r_cent[1:] * r_cent[:-1]) / denom, -0.999, 0.999)
        sigma2 = var_r / self.dt
        mu = mean_r * (1.0 - phi) / self.dt + 0.5 * sigma2
        return ARGBM(mu=float(mu), sigma=float(mx.sqrt(sigma2)),
                     phi=float(phi), s0=self.s0, dt=self.dt)

    def log_likelihood(self, returns) -> mx.array:
        """Conditional AR(1) likelihood plus the stationary initial term."""
        c = self._drift()
        sc = self._scale()
        base = self.sigma * (self.dt**0.5)
        z0 = (returns[..., 0] - self._stat_mean()) / base
        ll0 = -0.5 * (mx.log(2.0 * mx.pi) + 2.0 * mx.log(base) + z0 * z0)
        z = (returns[..., 1:] - c - self.phi * returns[..., :-1]) / sc
        ll_c = -0.5 * (mx.log(2.0 * mx.pi) + 2.0 * mx.log(sc) + z * z)
        return ll0 + mx.sum(ll_c, axis=-1)

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
            "ar1": autocorr(r, max_lag=1)[..., 0],
            "seed": mx.array(dataset.seed, dtype=mx.int64),
        }