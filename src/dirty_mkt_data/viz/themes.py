"""Color palettes for the plotly inspectors.

Each theme is a plain dataclass: background, grid, text, candle up/down
colors and volume colors. Follows the project's "no numpy in source" rule —
these are pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    name: str
    background: str
    plot_background: str
    grid: str
    text: str
    up: str            # bullish candle
    down: str          # bearish candle
    accent: str


def _vol_color(c: str, alpha: str = "45") -> str:
    """rgba() string with hex alpha suffix (plotly 8-digit hex alpha)."""
    return f"{c}{alpha}"


GHIBLI = Theme(
    name="ghibli",
    background="#F3EEDC",
    plot_background="#FBF8EE",
    grid="#E4DCC3",
    text="#3D3A2A",
    up="#5F9266",
    down="#D97757",
    accent="#8FB0C9",
)

SYNTHWAVE = Theme(
    name="synthwave",
    background="#241B2F",
    plot_background="#1B1424",
    grid="#3A2E52",
    text="#F2EFFF",
    up="#FF2E88",
    down="#22E4FF",
    accent="#B967FF",
)

VALORANT = Theme(
    name="valorant",
    background="#0F1923",
    plot_background="#141E2A",
    grid="#1F2F3B",
    text="#ECE8E1",
    up="#FF4655",
    down="#40C4A6",
    accent="#FFD166",
)

THEMES: dict[str, Theme] = {t.name: t for t in (GHIBLI, SYNTHWAVE, VALORANT)}

FONT_FAMILY = "JetBrains Mono"
MONO_SUFFIX_COMBO = "JetBrains Mono, Menlo, Monaco, Consolas, monospace"