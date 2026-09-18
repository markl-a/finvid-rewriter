"""Locate a system font with Traditional Chinese glyphs (for PIL overlays and matplotlib charts).

Override with FINVID_FONT=/path/to/font.ttf (Settings.font). Nothing is downloaded.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# (regular, bold-or-None) pairs, most preferred first
_CANDIDATES: list[tuple[str, str | None]] = [
    # Windows
    ("C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msjhbd.ttc"),      # Microsoft JhengHei
    ("C:/Windows/Fonts/NotoSansTC-VF.ttf", None),
    ("C:/Windows/Fonts/mingliu.ttc", None),
    # macOS
    ("/System/Library/Fonts/PingFang.ttc", None),
    ("/System/Library/Fonts/STHeiti Medium.ttc", None),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", None),
    # Linux (fonts-noto-cjk)
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc"),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", None),
    ("/usr/share/fonts/opentype/noto/NotoSansTC-Regular.otf", "/usr/share/fonts/opentype/noto/NotoSansTC-Bold.otf"),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", None),
]

HELP = (
    "No font with Traditional Chinese glyphs found. Set FINVID_FONT=/path/to/font.ttf|ttc, or install one: "
    "Windows ships Microsoft JhengHei (msjh.ttc); macOS ships PingFang; "
    "Linux: `apt install fonts-noto-cjk` (NotoSansCJK-Regular.ttc)."
)


def _override() -> str | None:
    env = os.environ.get("FINVID_FONT")
    if env:
        return env
    try:  # Settings may carry it from .env; imported lazily to avoid a cycle at module import
        from ..config import get_settings

        return get_settings().font
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=1)
def find_cjk_font() -> str:
    """Path to a regular-weight TTF/TTC/OTF with Traditional Chinese glyphs."""
    ov = _override()
    if ov:
        if Path(ov).exists():
            return str(Path(ov))
        raise FileNotFoundError(f"FINVID_FONT={ov} does not exist. {HELP}")
    for reg, _ in _CANDIDATES:
        if Path(reg).exists():
            return reg
    raise FileNotFoundError(HELP)


@lru_cache(maxsize=1)
def find_cjk_font_bold() -> str:
    """Bold variant if the platform has one next to the regular face, else the regular face."""
    reg = find_cjk_font()
    for r, b in _CANDIDATES:
        if r == reg and b and Path(b).exists():
            return b
    return reg


@lru_cache(maxsize=1)
def matplotlib_font_family() -> str:
    """Register the CJK font with matplotlib and return its family name (also sets rcParams)."""
    import matplotlib
    from matplotlib import font_manager

    path = find_cjk_font()
    font_manager.fontManager.addfont(path)
    name = font_manager.FontProperties(fname=path).get_name()
    matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False  # CJK fonts often lack U+2212
    return name
