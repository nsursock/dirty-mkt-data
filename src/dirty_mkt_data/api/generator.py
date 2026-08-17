"""Top-level ``Generator``: composes a ``Model`` (+ optional contamination)."""

from __future__ import annotations

from dataclasses import replace

import mlx.core as mx

from dirty_mkt_data.api.base import Contamination, Dataset, Model
from dirty_mkt_data.api.seeding import SeedContract


class Generator:
    """Deterministic synthetic-data factory.

    ``Generator(model, seed=42).sample(...)`` is reproducible byte-for-byte:
    the same seed and namespace always derive the same MLX random key.

    ``contamination`` is opt-in and applied last (P1 concrete layers).
    """

    def __init__(self, model: Model, seed: int = 42, contamination: Contamination | None = None):
        if not isinstance(model, Model):
            raise TypeError("model must be a dirty_mkt_data.api.base.Model")
        self.model = model
        self.seed = int(seed)
        self.contamination = contamination
        self._contract = SeedContract(self.seed)

    def _key(self, run_id: int) -> mx.array:
        return self._contract.stream(self.model.namespace, run_id)

    def sample(self, n_steps: int, n_paths: int = 1, run_id: int = 0) -> Dataset:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        dataset = self.model.sample(n_steps, n_paths=n_paths, key=self._key(run_id))
        dataset = replace(dataset, seed=self.seed)
        if self.contamination is not None:
            dataset = self.contamination.transform(
                dataset, key=self._contract.stream(self.contamination.namespace, run_id)
            )
        return dataset