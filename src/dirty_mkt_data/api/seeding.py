"""Deterministic seed contract.

Reproducibility guarantee: for a fixed integer ``seed`` and a fixed
``namespace``, ``SeedContract(seed).key(namespace)`` returns the *same* MLX
key forever. Callers must bump the version in a model's ``namespace``
whenever its sampling algorithm changes, exactly like a dataset schema
version.

Namespaces are folded via SHA-256 (process-stable, unlike ``hash()``) and
mixed through splitmix64 so distinct namespaces / seeds yield distinct,
uniformly-spaced 64-bit keys.
"""

from __future__ import annotations

import hashlib

import mlx.core as mx

_M64 = (1 << 64) - 1


def _fold_name(namespace: str) -> int:
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") & _M64


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return (x ^ (x >> 31)) & _M64


class SeedContract:
    """Deterministic mapping from (seed, namespace) to an MLX random key."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed) & _M64

    def digest(self, namespace: str) -> int:
        """Pure-Python 64-bit digest for ``namespace`` under this seed.

        Fully reproducible across processes and MLX versions; used to back
        ``key()``. Kept public so the contract can be asserted without
        touching MLX state.
        """
        salt = _splitmix64(_fold_name(namespace))
        return _splitmix64(self.seed ^ salt)

    def key(self, namespace: str) -> mx.array:
        return mx.random.key(self.digest(namespace))

    def stream(self, namespace: str, index: int) -> mx.array:
        """Key for the ``index``-th stream under ``namespace``.

        Splitting keeps streams independent while staying deterministic.
        """
        base = self.key(namespace)
        count = max(index + 1, 2)
        return mx.random.split(base, count)[index]