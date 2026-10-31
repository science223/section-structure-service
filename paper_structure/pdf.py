"""PDF to plain text, exactly as the benchmark made its paper texts.

Source: paper_benchmark/src/extract_pdfs.py, read 2026-10-19: pypdfium2 ``get_text_range`` page by page, pages joined
with "\\n", then "\\r\\n" and "\\r" normalised to "\\n". The model was trained and evaluated on text made this way, so
a different extractor would feed it line breaks and spacing it has not seen.

``extract_pdf_pages`` returns the same text plus where each page starts in it (for page numbers in outlines and for
"open the PDF at this page" links); the text is built exactly as ``extract_pdf_text`` builds it.
"""

from __future__ import annotations

import bisect
from pathlib import Path

from ._install import NOT_ON_PYPI, install_hint


def _raw_pages(source: str | Path | bytes) -> list[str]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise ImportError(f"Reading PDFs needs pypdfium2: {install_hint('pdf', 'pypdfium2')}; {NOT_ON_PYPI}") from exc
    pdf = pdfium.PdfDocument(str(source) if isinstance(source, Path) else source)
    try:
        return [page.get_textpage().get_text_range() for page in pdf]
    finally:
        pdf.close()


def normalise(raw: str) -> str:
    return raw.replace("\r\n", "\n").replace("\r", "\n")


def join_pages(pages: list[str]) -> tuple[str, list[int]]:
    """The benchmark's text of these raw pages and the position in it where each page starts.
    Positions are mapped through the normalisation: every "\\r\\n" (also one formed by a page ending in "\\r" and the
    joining "\\n") loses its "\\r", so a raw position moves left by the number of such pairs before it."""
    raw = "\n".join(pages)
    starts_raw, pos = [], 0
    for p in pages:
        starts_raw.append(pos)
        pos += len(p) + 1
    pairs, i = [], raw.find("\r\n")
    while i != -1:
        pairs.append(i)
        i = raw.find("\r\n", i + 2)
    # raw index r keeps its place minus the pairs whose dropped "\r" (at s) lies before r
    starts = [r - bisect.bisect_left(pairs, r) for r in starts_raw]
    return normalise(raw), starts


def extract_pdf_text(source: str | Path | bytes) -> str:
    """Plain text of a PDF given as a path or as its bytes."""
    return normalise("\n".join(_raw_pages(source)))


def extract_pdf_pages(source: str | Path | bytes) -> tuple[str, list[int]]:
    """(the text extract_pdf_text returns, start position of every page in it); page i (0-based) is
    text[starts[i]:starts[i + 1]]."""
    return join_pages(_raw_pages(source))


def page_of(starts: list[int], pos: int) -> int:
    """1-based page number of a position in the text."""
    return max(1, bisect.bisect_right(starts, pos))
