"""P0 — seed contract: deterministic, versioned MLX key derivation."""

import os
import subprocess
import sys
from pathlib import Path

import mlx.core as mx

from dirty_mkt_data.api.seeding import SeedContract

NS = "dirty_mkt_data.core.gbm.v1"

_SRC = str(Path(__file__).resolve().parents[1] / "src")
_CODE = (
    "from dirty_mkt_data.api.seeding import SeedContract; "
    f"print(SeedContract(42).digest('{NS}'))"
)


def _run() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _CODE],
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONPATH": _SRC},
    )


def test_same_seed_same_key():
    a = SeedContract(42).key(NS)
    b = SeedContract(42).key(NS)
    assert bool(mx.all(a == b))


def test_different_seed_different_key():
    a = SeedContract(42).key(NS)
    b = SeedContract(43).key(NS)
    assert not bool(mx.all(a == b))


def test_different_namespace_different_key():
    a = SeedContract(42).key(NS)
    b = SeedContract(42).key(NS + ".other")
    assert not bool(mx.all(a == b))


def test_key_is_uint_pair():
    k = SeedContract(42).key(NS)
    assert k.shape == (2,)
    assert k.dtype in (mx.uint32, mx.uint64)


def test_streams_are_independent_and_deterministic():
    c = SeedContract(7)
    s0a = c.stream(NS, 0)
    s1a = c.stream(NS, 1)
    s0b = SeedContract(7).stream(NS, 0)
    assert bool(mx.all(s0a == s0b))
    assert not bool(mx.all(s0a == s1a))


def test_idempotent_across_processes():
    out1 = _run()
    out2 = _run()
    assert out1.stdout == out2.stdout
    assert out1.returncode == 0
    assert int(out1.stdout) == SeedContract(42).digest(NS)


def test_key_matches_digest():
    k = SeedContract(42).key(NS)
    d = SeedContract(42).digest(NS)
    assert int(k[0]) == (d >> 32) & 0xFFFFFFFF
    assert int(k[1]) == d & 0xFFFFFFFF