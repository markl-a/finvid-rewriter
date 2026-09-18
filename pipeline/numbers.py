"""Number provenance gate ($0, pure Python).

The Pass B prompt tells the model to use only figures from the transcript. That is a promise,
not a check. This module extracts every number the transcript actually says — digits and
Chinese numerals alike (「一萬五」「五千八百億」「四成」「百分之四十」) — so a chart value that
never appears in the source can be dropped instead of drawn.
"""
from __future__ import annotations

import re

_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_SMALL = {"十": 10, "百": 100, "千": 1000}
_BIG = {"萬": 10_000, "万": 10_000, "億": 100_000_000, "亿": 100_000_000}
UNIT_SCALE = {"萬": 10_000, "万": 10_000, "億": 100_000_000, "亿": 100_000_000}

_RUN = re.compile(
    r"(?:百分之)?"
    r"[0-9０-９][0-9０-９,.]*(?:[萬万億亿][0-9０-９]?(?:[萬万億亿])?)?"
    r"|(?:百分之)?[零〇一二兩两三四五六七八九十百千萬万億亿點点]+"
)
_ARABIC = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def _fullwidth(s: str) -> str:
    return s.translate(str.maketrans("０１２３４５６７８９", "0123456789"))


def _parse_small(s: str) -> float:
    """Chinese numeral below 萬: 四十 -> 40, 五千八百 -> 5800, 五千八 -> 5800 (colloquial), 十五 -> 15."""
    total = 0.0
    cur = 0.0
    last_unit = 0
    for ch in s:
        if ch in _DIGITS:
            cur = cur * 10 + _DIGITS[ch] if cur >= 10 else _DIGITS[ch]
        elif ch in _SMALL:
            unit = _SMALL[ch]
            total += (cur if cur else 1) * unit
            cur = 0.0
            last_unit = unit
    if cur:
        # 五千八 = 5800, 三百五 = 350, 四十五 = 45
        total += cur * (last_unit / 10) if last_unit and cur < 10 and last_unit > 10 else cur
    return total


def _parse_cn(s: str) -> float | None:
    """Chinese numeral with 萬/億 sections and optional 點 decimals."""
    if "點" in s or "点" in s:
        head, _, tail = s.replace("点", "點").partition("點")
        base = _parse_cn(head) if head else 0.0
        if base is None:
            return None
        frac = "".join(str(_DIGITS[c]) for c in tail if c in _DIGITS)
        return float(f"{int(base)}.{frac}") if frac else base
    if not any(c in _DIGITS or c in _SMALL or c in _BIG for c in s):
        return None
    total = 0.0
    rest = s
    for big_ch, big in (("億", 1e8), ("亿", 1e8), ("萬", 1e4), ("万", 1e4)):
        if big_ch in rest:
            head, _, rest = rest.partition(big_ch)
            total += (_parse_small(head) if head else 1) * big
            # 一萬五 = 15000: a bare trailing digit after 萬/億 means the next unit down
            if rest and all(c in _DIGITS for c in rest) and len(rest) == 1:
                return total + _DIGITS[rest] * big / 10
    return total + _parse_small(rest)


def _split_cn(run: str) -> list[str]:
    """STT glues neighbouring numbers together: 「三萬二三萬四」 is two numbers. Cut a run when a
    second 萬/億 of the same magnitude appears, or when two bare digits follow 萬 (「三萬二|三」)."""
    chunks: list[str] = []
    cur = ""
    seen_big = 0.0
    after_big = after_big_bare = False
    for ch in run:
        if ch in _BIG:
            if seen_big and _BIG[ch] >= seen_big:
                chunks.append(cur)
                cur = ""
            seen_big = _BIG[ch]
            after_big, after_big_bare = True, False
        elif ch in _DIGITS:
            if after_big_bare:
                chunks.append(cur)
                cur, seen_big = "", 0.0
                after_big_bare = False
            elif after_big:
                after_big_bare = True
            after_big = False
        else:
            after_big = after_big_bare = False
        cur += ch
    if cur:
        chunks.append(cur)
    return chunks


def _parse_run(run: str) -> list[float]:
    """One token -> the values it can mean (percent forms yield the bare percentage)."""
    run = _fullwidth(run)
    pct = run.startswith("百分之")
    if pct:
        run = run[3:]
    vals: list[float] = []
    if run and run[0].isdigit():
        m = _ARABIC.match(run.replace(",", ""))
        if not m:
            return vals
        v = float(m.group())
        tail = run.replace(",", "")[m.end():]
        vals.append(v)
        if tail:
            big = UNIT_SCALE.get(tail[0])
            if big:
                scaled = v * big
                extra = tail[1:2]
                if extra.isdigit():  # 1萬5 = 15000
                    scaled += int(extra) * big / 10
                vals.append(scaled)
    else:
        for chunk in _split_cn(run):
            v = _parse_cn(chunk)
            if v is not None:
                vals.append(v)
    return vals


def number_tokens(text: str) -> list[tuple[str, list[float]]]:
    """Each number-ish token in the text with the value(s) it can mean:
    「1.5萬」 -> [1.5, 15000], 「四成」 -> [40], 「三萬二三萬四」 -> [32000, 34000]."""
    out: list[tuple[str, list[float]]] = []
    for m in _RUN.finditer(text):
        vals = _parse_run(m.group())
        if vals:
            out.append((m.group(), vals))
    for m in re.finditer(r"([一二兩两三四五六七八九十]|[0-9]+)成(?![本交績效])", text):
        v = _parse_run(m.group(1))
        if v:
            out.append((m.group(), [v[0] * 10]))
    return out


def numbers_in(text: str) -> set[float]:
    """Every numeric value the text states, in absolute terms (1.5萬 -> 15000, 四成 -> 40)."""
    return {v for _, vals in number_tokens(text) for v in vals}


def unstated_numbers(text: str, source: set[float], *, ignore: set[float] = frozenset()) -> list[str]:
    """Tokens in `text` none of whose readings appear in `source` (years and the like go in `ignore`)."""
    bad: list[str] = []
    for tok, vals in number_tokens(text):
        if any(v in ignore for v in vals):
            continue
        if not any(value_stated(v, "", source) for v in vals):
            bad.append(tok)
    return bad


def value_stated(value: float, unit: str, source: set[float], rel_tol: float = 0.005) -> bool:
    """Does the transcript (or extracted data_points) state this chart value?
    Both the bare value and the unit-scaled value count: '36000 元' matches 「三萬六」 (36000),
    '5800 億' matches 「五千八百億」 (5.8e11) or the bare 「5800」."""
    targets = {float(value)}
    scale = UNIT_SCALE.get(unit.strip()[:1] if unit else "")
    if scale:
        targets.add(float(value) * scale)
    for t in targets:
        for s in source:
            if abs(s - t) <= max(1e-6, rel_tol * abs(t)):
                return True
    return False
