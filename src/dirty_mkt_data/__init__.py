"""Synthetic financial market-data generator. Apple MLX only."""

from dirty_mkt_data.api.base import Dataset, Model
from dirty_mkt_data.api.generator import Generator
from dirty_mkt_data.api.seeding import SeedContract

__all__ = ["Dataset", "Generator", "Model", "SeedContract"]
__version__ = "0.1.0"