"""Open documents with the Navigator (model output cached on disk) and list what looks wrong in their outlines, for a
person to look at: repeated ids, sections with no body although nothing is nested in them, very short sections,
headings the locator could not place, and the contents entries / page headers that were skipped.
  python scripts/scan_outlines.py <out.json> <file or folder> [<file or folder> ...] [--limit N]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from paper_structure.service import Navigator  # noqa: E402

SHORT = 40   # characters of body below which a leaf section is listed


def scan(nav: Navigator, f: Path) -> dict:
    t = time.time()
    od = nav.open(f)
    secs = od.document.sections
    flags = {"repeated_ids": [s.id for s in secs if "#" in s.id], "empty_leaves": [], "short_leaves": []}
    for i, s in enumerate(secs):
        has_child = i + 1 < len(secs) and secs[i + 1].depth > s.depth
        body = od.text[s.char_start + len(s.title):s.char_end].strip()
        if has_child:
            continue
        if not body:
            flags["empty_leaves"].append(s.id)
        elif len(body) < SHORT:
            flags["short_leaves"].append(f"{s.id}: {' '.join(body.split())[:SHORT]}")
    return {"file": f.name, "chars": len(od.text), "seconds": round(time.time() - t, 1), "from_cache": od.from_cache,
            "sections": len(secs), "max_depth": max((s.depth for s in secs), default=0),
            "unlocated": [u.predicted_title[:60] for u in od.document.unlocated],
            "contents_skipped": len(od.toc_dropped), "repeats_skipped": len(od.repeats_dropped), **flags}


def main(argv: list[str]) -> int:
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else 0
    args = [a for i, a in enumerate(argv) if a != "--limit" and (i == 0 or argv[i - 1] != "--limit")]
    out, sources = Path(args[0]), [Path(a) for a in args[1:]]
    files = []
    for s in sources:
        found = sorted(p for p in (s.glob("*") if s.is_dir() else [s]) if p.suffix.lower() in (".pdf", ".txt"))
        files += found[:limit] if limit else found
    nav = Navigator()
    rows = []
    for f in files:
        try:
            r = scan(nav, f)
        except Exception as exc:  # noqa: BLE001 - a document that fails is a finding too
            r = {"file": f.name, "error": f"{type(exc).__name__}: {exc}"[:300]}
        rows.append(r)
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        flag = "" if "error" not in r else " ERROR " + r["error"]
        if "error" not in r:
            flag = (f" ids#{len(r['repeated_ids'])} empty{len(r['empty_leaves'])} short{len(r['short_leaves'])}"
                    f" unloc{len(r['unlocated'])} toc{r['contents_skipped']} rep{r['repeats_skipped']}")
        print(f"{f.name[:40]:40s} {r.get('sections', '-'):>4} secs {r.get('seconds', '-'):>6}s{flag}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
