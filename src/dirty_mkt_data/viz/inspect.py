"""Interactive plotly inspection of generated market data.

The MLX->Python list conversion happens *only* here, at the rendering
boundary (plotly needs host-side data); the rest of the library stays pure
MLX. Candle + volume subplot in an arbitrary ``Theme``.
"""

from __future__ import annotations

from datetime import date, timedelta

import mlx.core as mx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dirty_mkt_data.viz.ohlcv import OHLCV
from dirty_mkt_data.viz.themes import MONO_SUFFIX_COMBO, Theme

FIG_WIDTH = 1366
FIG_HEIGHT = 820


def _dates(n_steps: int, start: str = "2022-01-03") -> list[str]:
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n_steps)]


def _rgba(color: str, alpha: float) -> str:
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def candle_figure(
    ohlcv: OHLCV,
    theme: Theme,
    path: int = 0,
    title: str | None = None,
    show_volume: bool = True,
    width: int = FIG_WIDTH,
    height: int = FIG_HEIGHT,
    font_family: str = MONO_SUFFIX_COMBO,
) -> go.Figure:
    """Candlestick (+ optional volume) figure for one generated path."""
    n_steps = ohlcv.closes.shape[-1]
    dates = _dates(n_steps)

    op = ohlcv.opens[path].tolist()
    hi = ohlcv.highs[path].tolist()
    lo = ohlcv.lows[path].tolist()
    cl = ohlcv.closes[path].tolist()
    vo = ohlcv.vols[path].tolist()

    rows, row_heights = (2, [0.75, 0.25]) if show_volume else (1, [1.0])
    fig = make_subplots(
        rows=rows,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=op,
            high=hi,
            low=lo,
            close=cl,
            name="OHLC",
            increasing_line_color=theme.up,
            increasing_fillcolor=theme.up,
            decreasing_line_color=theme.down,
            decreasing_fillcolor=theme.down,
        ),
        row=1,
        col=1,
    )

    if show_volume:
        colors = [
            _rgba(theme.up, 0.45) if d > 0 else
            _rgba(theme.down, 0.45) if d < 0 else
            _rgba(theme.grid, 0.6)
            for d in mx.sign(ohlcv.closes[path] - ohlcv.opens[path]).tolist()
        ]
        fig.add_trace(
            go.Bar(x=dates, y=vo, marker_color=colors, name="Volume"),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=title or f"{theme.name.upper()} — synthetic GBM",
        paper_bgcolor=theme.background,
        plot_bgcolor=theme.plot_background,
        font=dict(family=font_family, color=theme.text, size=12),
        width=width,
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=48, r=24, t=64, b=32),
    )
    fig.update_xaxes(gridcolor=theme.grid, rangeslider_visible=False)
    fig.update_yaxes(gridcolor=theme.grid, zerolinecolor=theme.grid, row=1, col=1)
    if show_volume:
        fig.update_yaxes(gridcolor=theme.grid, showticklabels=True, row=2, col=1)

    return fig