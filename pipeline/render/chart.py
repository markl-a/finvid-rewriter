"""ChartSpec -> PNG with matplotlib. Charts are always re-drawn from the extracted numbers,
never screenshotted from the source video (legality requirement).

Style: one accent colour, a muted second series if needed, no top/right spines, light y-grid,
value labels on the marks, CJK font. Sized to stay legible when the 1080-wide video is viewed on a phone.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..models import ChartSpec  # noqa: E402
from .fonts import matplotlib_font_family  # noqa: E402

ACCENT = "#2563EB"
SECOND = "#F59E0B"
INK = "#1E293B"
INK_MUTED = "#64748B"
GRID = "#E2E8F0"
BG = "#FFFFFF"
DPI = 200


def fmt_value(v: float, unit: str = "") -> str:
    s = f"{v:,.0f}" if abs(v) >= 1000 else f"{v:g}"
    return f"{s}{unit}"


def render_chart(spec: ChartSpec, out_png: Path, width_px: int = 960, height_px: int = 720) -> Path:
    family = matplotlib_font_family()
    series = [s for s in spec.series if s.points][:2]  # at most two series: accent + muted
    if not series:
        raise ValueError("ChartSpec has no data points")

    fig, ax = plt.subplots(figsize=(width_px / DPI, height_px / DPI), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    labels = [p.label for p in series[0].points]
    x = list(range(len(labels)))
    colors = [ACCENT, SECOND]
    unit = next((p.unit for s in series for p in s.points if p.unit), "")
    txt = {"fontsize": 10, "color": INK, "family": family}

    if spec.type == "line":
        for s, c in zip(series, colors):
            ys = [p.value for p in s.points]
            ax.plot(x[:len(ys)], ys, color=c, linewidth=2.5, marker="o", markersize=7, label=s.name,
                    solid_capstyle="round", zorder=3)
            for xi, p in zip(x, s.points):
                ax.annotate(fmt_value(p.value, p.unit or unit), (xi, p.value), textcoords="offset points",
                            xytext=(0, 9), ha="center", zorder=4, **txt)
    else:
        n = len(series)
        width = 0.6 if n == 1 else 0.36
        for k, (s, c) in enumerate(zip(series, colors)):
            offs = 0 if n == 1 else (k - 0.5) * (width + 0.04)
            xs = [xi + offs for xi in x[:len(s.points)]]
            ys = [p.value for p in s.points]
            bars = ax.bar(xs, ys, width=width, color=c, label=s.name, zorder=3)
            for b, p in zip(bars, s.points):
                h = b.get_height()
                va, dy = ("bottom", 3) if h >= 0 else ("top", -3)
                ax.annotate(fmt_value(p.value, p.unit or unit), (b.get_x() + b.get_width() / 2, h),
                            textcoords="offset points", xytext=(0, dy), ha="center", va=va, zorder=4, **txt)
        ax.axhline(0, color=INK_MUTED, linewidth=0.8, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, color=INK, family=family)
    ax.tick_params(axis="y", labelsize=9.5, colors=INK_MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    if spec.y_label:
        ax.set_ylabel(spec.y_label, fontsize=9.5, color=INK_MUTED, family=family)
    ax.set_title(spec.title, fontsize=13, color=INK, family=family, fontweight="bold", loc="left", pad=12)
    # headroom for the value labels
    ymin, ymax = ax.get_ylim()
    span = (ymax - ymin) or 1.0
    ax.set_ylim(ymin if ymin < 0 else 0, ymax + span * 0.12)
    if len(series) > 1:
        ax.legend(frameon=False, loc="upper right", prop={"family": family, "size": 9.5})

    fig.tight_layout(pad=0.6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out_png
