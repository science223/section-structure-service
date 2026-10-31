"""Numbering-based section-depth heuristics.

New TASK X adapter, based only on the numbering rules in the task brief. It is
not copied from the benchmark and intentionally does not infer unnumbered trees.
"""

from __future__ import annotations

import re


_NUMBERING_RE = re.compile(
    r"^\s*((?:\d+|[A-Z])(?:\.\d+)*)(?:\.(?=\s|$)|(?=\s|$))"
)


def numbering_token(title: str) -> str | None:
    """Return a leading ``3``, ``3.1``, ``A.2``, etc., when present."""
    match = _NUMBERING_RE.match(title)
    return match.group(1) if match else None


def infer_depth(title: str) -> int:
    """Infer depth from numbering; every unnumbered title has depth 1."""
    token = numbering_token(title)
    return token.count(".") + 1 if token else 1


_ROMAN_RE = re.compile(r"^\s*([IVX]{1,6})\.(?=\s|$)")   # "$": a heading cut after its number ("B.") counts too
_LETTER_RE = re.compile(r"^\s*([A-Z])\.(?=\s|$)")
_ITEM_RE = re.compile(r"^\s*(\d{1,2})\)(?=\s|$)")
# the whole title (fullmatch): "Supplemental Results" between two lettered subsections is not back matter
_TOP_LEVEL_RE = re.compile(r"\s*(?:abstract|index terms|keywords|references|bibliography|acknowledg(?:e)?ments?|"
                           r"appendix(?:\s+[A-Z0-9]{1,3})?|appendices|biograph(?:y|ies)|author biographies|nomenclature|"
                           r"supplementary\s+(?:material|materials|information))\s*[.:]?\s*", re.IGNORECASE)
_ROMAN_DIGITS = (("X", 10), ("IX", 9), ("V", 5), ("IV", 4), ("I", 1))


def _roman(n: int) -> str:
    out = ""
    for sym, val in _ROMAN_DIGITS:
        while n >= val:
            out, n = out + sym, n - val
    return out


_ROMAN_VALUE = {_roman(n): n for n in range(1, 40)}


def outline_numbering(titles: list[str]) -> list[tuple[str | None, int]]:
    """(printed number or None, depth) for each heading of one document, in order.

    Printed numbers ("3", "3.1", "A.2") give the depth by their dots, and unnumbered headings have depth 1, exactly
    as numbering_token / infer_depth. The one exception is the IEEE layout, recognised when at least two headings
    are Roman-numeral sections in order ("I. INTRODUCTION", "II. ..."; one missing number allowed): there a lettered
    heading ("A. Naive RAG") belongs to the last Roman section ("II.A", depth 2), "1)" to the last letter
    ("II.A.1", depth 3), and an unnumbered heading inside a Roman section sits one level below the last numbered
    heading, until a front- or back-matter heading (References, Appendix, ...) ends the Roman sections. Letters
    restart in every IEEE section, so without this every "A." would be depth 1 and the ids "A", "A#2", "A#3"."""
    roman_at, last = set(), 0
    for i, t in enumerate(titles):
        m = _ROMAN_RE.match(t)
        value = _ROMAN_VALUE.get(m.group(1)) if m else None
        if value is not None and last < value <= last + 2:
            roman_at.add(i)
            last = value
    ieee = len(roman_at) >= 2
    out: list[tuple[str | None, int]] = []
    roman = letter = None
    level = 0   # depth of the last numbered heading inside the current Roman section
    for i, t in enumerate(titles):
        if ieee:
            if i in roman_at:
                roman, letter, level = _ROMAN_RE.match(t).group(1), None, 1
                out.append((roman, 1))
                continue
            if _TOP_LEVEL_RE.fullmatch(t):
                roman = letter = None
            elif roman is not None:
                if m := _LETTER_RE.match(t):
                    letter, level = m.group(1), 2
                    out.append((f"{roman}.{letter}", 2))
                    continue
                if m := _ITEM_RE.match(t):
                    level = 3 if letter else 2
                    out.append((f"{roman}.{letter}.{m.group(1)}" if letter else f"{roman}.{m.group(1)}", level))
                    continue
                if numbering_token(t) is None:
                    out.append((None, level + 1))
                    continue
        token = numbering_token(t)
        out.append((token, token.count(".") + 1 if token else 1))
    return out


SLUG_WORDS = 6


def section_id_base(title: str, ordinal: int) -> str:
    """Prefer the printed section number; otherwise make a readable slug from the first SLUG_WORDS words
    (a predicted heading can run into body text; build_document still makes duplicates unique)."""
    token = numbering_token(title)
    if token:
        return token
    words = re.sub(r"[^a-z0-9]+", " ", title.casefold()).split()
    slug = "-".join(words[:SLUG_WORDS])
    return slug or f"section-{ordinal}"
