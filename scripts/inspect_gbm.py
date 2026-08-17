"""Render themed OHLCV charts of a synthetic GBM sample (config-driven).

All parameters live in configs/inspect.yaml; CLI flags override config values.

Usage:
    PYTHONPATH=src venv/bin/python scripts/inspect_gbm.py
    PYTHONPATH=src venv/bin/python scripts/inspect_gbm.py --config configs/inspect.yaml --steps 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mlx.core as mx  # noqa: E402

from dirty_mkt_data import Generator  # noqa: E402
from dirty_mkt_data.core.gbm import GBM  # noqa: E402
from dirty_mkt_data.viz.inspect import candle_figure  # noqa: E402
from dirty_mkt_data.viz.ohlcv import from_dataset  # noqa: E402
from dirty_mkt_data.viz.themes import THEMES  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "inspect.yaml"


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as fh:
        return yaml.safe_load(fh)


def _build_parser(defaults: dict) -> argparse.ArgumentParser:
    gen = defaults["generator"]
    params = gen["params"]
    viz = defaults["viz"]
    png = viz["png"]
    ap = argparse.ArgumentParser(description="Config-driven GBM OHLCV renderer")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--steps", type=int, default=gen["n_steps"])
    ap.add_argument("--n-paths", type=int, default=gen["n_paths"])
    ap.add_argument("--seed", type=int, default=gen["seed"])
    ap.add_argument("--sigma", type=float, default=params["sigma"])
    ap.add_argument("--mu", type=float, default=params.get("mu"))
    ap.add_argument("--s0", type=float, default=params["s0"])
    ap.add_argument("--dt-days", type=int, default=params["dt_days"])
    ap.add_argument("--path", type=int, default=gen["path"])
    ap.add_argument("--theme", action="append", default=viz["themes"],
                    help="palette(s); repeatable; overrides config theme list")
    ap.add_argument("--base-volume", type=float, default=defaults["ohlcv"]["base_volume"])
    ap.add_argument("--exc-scaling", type=float, default=defaults["ohlcv"]["exc_scaling"])
    ap.add_argument("--vol-noise", type=float, default=defaults["ohlcv"]["vol_noise"])
    ap.add_argument("--out", type=Path, default=ROOT / viz["out_dir"])
    ap.add_argument("--png", dest="png_enabled", action="store_true", default=png["enabled"])
    ap.add_argument("--no-png", dest="png_enabled", action="store_false")
    ap.add_argument("--html", dest="html_enabled", action="store_true", default=viz["html"]["enabled"])
    ap.add_argument("--scale", type=float, default=png["scale"])
    ap.add_argument("--open", dest="open_after", action="store_true", default=viz["open_after"])
    ap.add_argument("--no-open", dest="open_after", action="store_false")
    return ap


def main() -> int:
    defaults = _load_config(DEFAULT_CONFIG)
    args = _build_parser(defaults).parse_args()

    params = defaults["generator"]["params"]
    mu = args.mu
    if mu is None and params.get("log_martingale_mu", False):
        mu = 0.5 * args.sigma**2
    if mu is None:
        mu = 0.0

    dt = 1.0 / args.dt_days
    model = GBM(mu=mu, sigma=args.sigma, s0=args.s0)
    ds = Generator(model, seed=args.seed).sample(args.steps, n_paths=args.n_paths)
    ohlcv = from_dataset(
        ds,
        sigma=args.sigma,
        dt=dt,
        base_volume=args.base_volume,
        exc_scaling=args.exc_scaling,
        vol_noise=args.vol_noise,
        key=mx.random.key(args.seed + defaults["ohlcv"]["key_seed_offset"]),
    )

    themes = [THEMES[t] for t in args.theme]
    viz_cfg = defaults["viz"]
    w, h = viz_cfg["width"], viz_cfg["height"]
    args.out.mkdir(parents=True, exist_ok=True)
    files = []
    for theme in themes:
        title = (f"{viz_cfg['title_prefix']} {theme.name.upper()} · "
                 f"sigma={args.sigma} · n={args.steps} · seed={args.seed}")
        fig = candle_figure(
            ohlcv,
            theme,
            path=args.path,
            title=title,
            width=w,
            height=h,
            font_family=viz_cfg.get("font", "JetBrains Mono"),
        )
        png = args.out / f"gbm_ohlcv_n{args.steps}_{theme.name}.png"
        if args.png_enabled:
            fig.write_image(png, width=w, height=h, scale=args.scale)
            files.append(png)
        if args.html_enabled:
            html = args.out / f"gbm_ohlcv_n{args.steps}_{theme.name}.html"
            fig.write_html(html, include_plotlyjs="cdn", config={"scrollZoom": True})
            files.append(html)

    print(f"saved {len(files)} file(s) to {args.out}/")
    for f in files:
        print(f"  {f}")
    if args.open_after:
        import webbrowser

        for f in files:
            webbrowser.open(f.absolute().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())