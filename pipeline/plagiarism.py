"""Automated "rewrite, don't copy" gate. Pure functions, no I/O, $0.

Two complementary signals, both computed on a normalised string (whitespace and
punctuation stripped, only CJK + alphanumerics kept):

- ``ngram_overlap``: share of the script's character n-grams that also occur in the
  source. Catches diffuse copying (many short phrases lifted verbatim).
- ``longest_common_substring``: length of the longest run of characters shared with
  the source. Catches one long sentence lifted verbatim even when the rest is fresh.

Numbers and proper nouns ("台北市", "2.5%", "央行") legitimately overlap, which is why
the thresholds (Settings.plagiarism_*) are not zero: with n=6 a 6-char window has to
match exactly, so a shared "2.5%" or "台北市" alone never trips the gate, but a lifted
clause like "房價所得比已經來到" does.
"""
from __future__ import annotations

import re

# keep CJK unified ideographs, fullwidth/halfwidth digits+latin, and ASCII alphanumerics
_KEEP = re.compile(r"[^0-9A-Za-z一-鿿㐀-䶿０-９Ａ-Ｚａ-ｚ]+")


def normalise(text: str) -> str:
    """Strip whitespace/punctuation so that '房價，漲了 3%' and '房價漲了3%' compare equal."""
    return _KEEP.sub("", text or "").lower()


def _ngrams(s: str, n: int) -> set[str]:
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def ngram_overlap(script: str, source: str, n: int) -> float:
    """Fraction of the script's character n-grams that appear anywhere in the source.

    Returns 0.0 when the script is shorter than n characters (nothing to compare).
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    a = normalise(script)
    b = normalise(source)
    if len(a) < n:
        return 0.0
    src = _ngrams(b, n)
    grams = [a[i:i + n] for i in range(len(a) - n + 1)]
    hits = sum(1 for g in grams if g in src)
    return hits / len(grams)


def longest_common_substring(a: str, b: str) -> int:
    """Length (in normalised characters) of the longest substring shared by a and b.

    Classic two-row DP, O(len(a) * len(b)); scripts are < 300 chars and a segment of
    transcript < 3k chars, so this is microseconds.
    """
    x = normalise(a)
    y = normalise(b)
    if not x or not y:
        return 0
    if len(x) > len(y):  # iterate the shorter string in the inner loop
        x, y = y, x
    best = 0
    prev = [0] * (len(x) + 1)
    for j in range(1, len(y) + 1):
        cur = [0] * (len(x) + 1)
        yj = y[j - 1]
        for i in range(1, len(x) + 1):
            if x[i - 1] == yj:
                v = prev[i - 1] + 1
                cur[i] = v
                if v > best:
                    best = v
        prev = cur
    return best


def longest_common_substring_text(a: str, b: str) -> str:
    """The actual longest shared substring (normalised form), for feeding back to the LLM."""
    x = normalise(a)
    y = normalise(b)
    if not x or not y:
        return ""
    best = 0
    end_i = 0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        xi = x[i - 1]
        for j in range(1, len(y) + 1):
            if xi == y[j - 1]:
                v = prev[j - 1] + 1
                cur[j] = v
                if v > best:
                    best = v
                    end_i = i
        prev = cur
    return x[end_i - best:end_i]


def check(script_text: str, source_text: str, *, n: int, max_overlap: float,
          max_lcs: int) -> tuple[bool, float, int]:
    """Return (ok, overlap, lcs). ok is False if either metric exceeds its threshold."""
    overlap = ngram_overlap(script_text, source_text, n)
    lcs = longest_common_substring(script_text, source_text)
    ok = overlap <= max_overlap and lcs <= max_lcs
    return ok, overlap, lcs
