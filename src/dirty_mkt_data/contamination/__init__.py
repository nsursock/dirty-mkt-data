"""contamination package: opt-in data-quality defects applied *last*.

Concrete layers (P1):
    gaps.py   — missing bars / exchange outages
    noise.py  — microstructure noise, bad ticks, duplicate timestamps
    clock.py  — timestamp skew / misalignment

Each layer subclasses ``dirty_mkt_data.api.base.Contamination`` and receives
a ``key`` from the ``Generator`` so every defect is also seed-reproducible.
"""

from dirty_mkt_data.api.base import Contamination

__all__ = ["Contamination"]