"""Old vs new contents-page rule on a folder of texts: the model runs once per text (cached); both rules then work
on the same predicted headings. Used 2026-10-20 on 26 survey / technical-report texts (none is a benchmark test
paper: their arXiv ids were checked against every AKC id); results in docs/contents_rule_20261020/.
    git show 044b940:paper_structure/service.py > old_service.py
    python scripts/compare_contents_rule.py old_service.py <folder of .txt> <out.json>
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from paper_structure.service import Navigator, drop_toc_entries as new_rule  # noqa: E402

spec = importlib.util.spec_from_file_location("paper_structure.service_old", sys.argv[1])
old = importlib.util.module_from_spec(spec)
sys.modules["paper_structure.service_old"] = old
spec.loader.exec_module(old)

texts = sorted(Path(sys.argv[2]).glob("*.txt"))
out = Path(sys.argv[3])
nav = Navigator()
rows = []
for f in texts:
    t = time.time()
    od = nav.open(f)
    secs_t = time.time() - t
    a, a_drop = old.drop_toc_entries(od.text, od.headings)
    b, b_drop = new_rule(od.text, od.headings)
    sa = [(s.id, s.char_start) for s in a.sections]
    sb = [(s.id, s.char_start) for s in b.sections]
    row = {"file": f.name, "chars": len(od.text), "model_s": round(secs_t, 1), "from_cache": od.from_cache,
           "old_sections": len(sa), "new_sections": len(sb), "old_dropped": len(a_drop), "new_dropped": len(b_drop),
           "old_dup_ids": sum("#" in i for i, _ in sa), "new_dup_ids": sum("#" in i for i, _ in sb),
           "same": sa == sb,
           "only_old": [x for x in sa if x not in sb][:30], "only_new": [x for x in sb if x not in sa][:30]}
    rows.append(row)
    print(json.dumps({k: row[k] for k in ("file", "chars", "model_s", "old_sections", "new_sections", "old_dropped",
                                          "new_dropped", "old_dup_ids", "new_dup_ids", "same")}, ensure_ascii=False),
          flush=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print("done", sum(r["same"] for r in rows), "of", len(rows), "unchanged")
