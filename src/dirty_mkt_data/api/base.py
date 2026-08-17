"""Common interfaces shared by every generator.

The `Model` protocol is intentionally small so every generator (classic,
regime-switching, contamination-aware) is swappable through one pipeline.

    Model
      ├── fit(real_data) -> Model          # calibrate params from observed returns
      ├── sample(n_steps, n_paths, key)    # draw synthetic log-returns
      ├── log_likelihood(returns)          # eval synthetic series under the model
      └── diagnostics(dataset)             # stylized-fact stats, as MLX arrays
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class Dataset:
    """A generated market-data sample.

    All tensors are MLX arrays of shape ``(n_paths, n_steps)`` (time on the
    last axis). ``seed`` records the seed used for reproducibility.
    """

    prices: mx.array
    returns: mx.array
    n_paths: int
    n_steps: int
    seed: int = 0


class Model(abc.ABC):
    """Abstract generator interface. Subclasses own a versioned namespace."""

    namespace: str = "dirty_mkt_data.model.abstract.v0"

    def __init__(self, **params) -> None:
        self.params = dict(params)

    @abc.abstractmethod
    def sample(self, n_steps: int, n_paths: int = 1, key=None) -> Dataset:
        """Draw ``n_paths`` independent paths of ``n_steps`` log-returns."""

    def fit(self, returns) -> "Model":
        """Calibrate parameters from observed returns. Default: identity."""
        return self

    @abc.abstractmethod
    def log_likelihood(self, returns) -> mx.array:
        """Per-path total log-likelihood of ``returns`` under the model."""

    @abc.abstractmethod
    def diagnostics(self, dataset: Dataset) -> dict[str, mx.array]:
        """Stylized-fact statistics for the sample, as MLX arrays."""


class Contamination(abc.ABC):
    """Opt-in defect layer applied *last*, on top of a clean series (P1+)."""

    namespace: str = "dirty_mkt_data.contamination.abstract.v0"

    @abc.abstractmethod
    def transform(self, dataset: Dataset, key=None) -> Dataset:
        """Return a modified dataset (gaps, noise, clock skew, ...)."""