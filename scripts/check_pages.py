"""Check page starts on real PDFs: extract_pdf_pages must give exactly extract_pdf_text's text, and every page's own
text must start at its page start.  python scripts/check_pages.py <folder with PDFs>"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_structure.pdf import _raw_pages, extract_pdf_pages, extract_pdf_text, normalise  # noqa: E402


def main() -> int:
    bad = 0
    pdfs = sorted(Path(sys.argv[1]).glob("*.pdf"))
    for f in pdfs:
        text, starts = extract_pdf_pages(f)
        same = text == extract_pdf_text(f)
        pages = _raw_pages(f)
        aligned = all(text[s:].startswith(normalise(p)) for s, p in zip(starts, pages))
        ok = same and aligned and len(starts) == len(pages)
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {f.name}: {len(pages)} pages, {len(text)} chars, same text {same}, aligned {aligned}")
    print(f"{len(pdfs) - bad}/{len(pdfs)} ok")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
