"""Outline (id, depth, start) of every .txt and .pdf in a folder, with the paper_structure package of a given checkout
(python scripts/dump_outlines.py <checkout> <folder of .txt> <out.json>; run it for two checkouts and compare).
The model output comes from the shared cache, so two checkouts see the same predicted headings."""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from paper_structure.service import Navigator  # noqa: E402

nav = Navigator()
out = {}
for f in sorted(p for p in Path(sys.argv[2]).glob("*") if p.suffix.lower() in (".txt", ".pdf")):
    od = nav.open(f)
    secs = od.document.sections
    out[f.name] = {"sections": [[s.id, s.depth, s.char_start] for s in secs],
                   "dup_ids": sum("#" in s.id for s in secs), "max_depth": max((s.depth for s in secs), default=0),
                   "contents": len(od.toc_dropped), "repeats": len(getattr(od, "repeats_dropped", []) or [])}
Path(sys.argv[3]).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print("wrote", len(out))
