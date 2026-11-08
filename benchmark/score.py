"""Score heading predictions against the 62-paper gold labels.

    python benchmark/score.py <predictions dir> [--gold benchmark/gold_labels_62.jsonl] [--all-chunks]

<predictions dir> holds one <paper_id>.jsonl per paper, one line per chunk: {"chunk_id": 3, "predicted_headings": [...]}.
A missing chunk counts as predicting nothing; a missing paper is an error.

The metric is the one behind the reported numbers: per chunk, the normalised heading set (lower case, whitespace
collapsed, punctuation and numbering kept, no fuzzy matching) is compared with the gold set; a predicted heading that
is a false positive in its chunk counts as correct when it equals a still-unmatched gold heading in the chunk before or
after (+-1-chunk tolerance). Contents-page chunks are skipped. True and false positives and false negatives are summed
over all chunks (micro P/R/F1); macro F1 averages per-paper F1 over papers with any gold or predicted heading.

By default only the main-table chunks are scored (in_main_table: the 11 chunks that some LLM API refused are left out
for every system); --all-chunks scores all 2,072.
"""
import argparse
import json
from pathlib import Path


def norm(s):
    return " ".join(s.lower().split())


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score_paper(chunks, preds):
    rows = []
    for c in chunks:
        gold = {}
        for h in c["gold"]:
            gold.setdefault(norm(h["heading"]), h["heading"])
        pred = {}
        for h in preds.get(c["chunk_id"], []):
            pred.setdefault(norm(h), h)
        rows.append({"chunk_id": c["chunk_id"], "gold": list(gold), "tp": len(gold.keys() & pred.keys()),
                     "fp_list": sorted(pred.keys() - gold.keys()), "fn": len(gold.keys() - pred.keys()),
                     "used": set(gold.keys() & pred.keys())})
    by_id = {r["chunk_id"]: r for r in rows}
    for r in rows:   # +-1-chunk tolerance, in chunk order, previous chunk first
        for h in list(r["fp_list"]):
            for d in (-1, 1):
                n = by_id.get(r["chunk_id"] + d)
                if n is None or n["fn"] <= 0 or h in n["used"] or h not in n["gold"]:
                    continue
                r["fp_list"].remove(h)
                r["tp"] += 1
                n["fn"] -= 1
                n["used"].add(h)
                break
    return (sum(r["tp"] for r in rows), sum(len(r["fp_list"]) for r in rows), sum(r["fn"] for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--gold", type=Path, default=Path(__file__).with_name("gold_labels_62.jsonl"))
    ap.add_argument("--all-chunks", action="store_true")
    a = ap.parse_args()
    tot, per_paper = [0, 0, 0], []
    for line in open(a.gold, encoding="utf-8"):
        paper = json.loads(line)
        f = a.predictions / f"{paper['paper_id']}.jsonl"
        if not f.exists():
            raise SystemExit(f"no predictions for {paper['paper_id']} ({f})")
        preds = {}
        for l in open(f, encoding="utf-8"):
            if l.strip():
                p = json.loads(l)
                preds[p["chunk_id"]] = p.get("predicted_headings") or []
        chunks = [c for c in paper["chunks"] if not c["is_toc"] and (a.all_chunks or c["in_main_table"])]
        tp, fp, fn = score_paper(chunks, preds)
        tot = [tot[0] + tp, tot[1] + fp, tot[2] + fn]
        if tp + fp + fn:
            per_paper.append(prf(tp, fp, fn)[2])
    p, r, f1 = prf(*tot)
    print(json.dumps({"papers": len(per_paper), "chunks": "all" if a.all_chunks else "main table",
                      "tp": tot[0], "fp": tot[1], "fn": tot[2], "micro_P": round(p, 4), "micro_R": round(r, 4),
                      "micro_F1": round(f1, 4), "macro_F1": round(sum(per_paper) / len(per_paper), 4)}))


if __name__ == "__main__":
    main()
