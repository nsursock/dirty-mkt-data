# dirty-mkt-data

Synthetic financial market-data generator. **Apple MLX only** (no NumPy).

## Setup

```bash
python -m venv venv
venv/bin/pip install -e ".[dev]"
```

## Tests

```bash
venv/bin/python -m pytest
```

## Inspect charts

```bash
PYTHONPATH=src venv/bin/python scripts/inspect_gbm.py
```

## Layout

| Path | Role |
|------|------|
| `src/dirty_mkt_data/` | Library code |
| `tests/` | Pytest suite |
| `configs/` | YAML configs |
| `scripts/` | CLI helpers |
| `docs/` | Notes and priority lists |
| `figures/` | Example OHLCV renders |
