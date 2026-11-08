"""Download the benchmark PDFs from arXiv (pinned versions) and check their SHA-256.

    python benchmark/fetch_pdfs.py <out dir> [--gold benchmark/gold_labels_62.jsonl]

Saves <paper_id>.pdf for the 61 arXiv papers, waiting 3 s between requests as arXiv asks. Already-present files with
the right hash are skipped. B7_GNN_Review is the journal version (AI Open, DOI in the gold file); download it from the
publisher (open access) and save it as B7_GNN_Review.pdf; this script then only checks its hash.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--gold", type=Path, default=Path(__file__).with_name("gold_labels_62.jsonl"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ok, bad, manual = 0, [], []
    for line in open(a.gold, encoding="utf-8"):
        r = json.loads(line)
        f = a.out / f"{r['paper_id']}.pdf"
        if not (f.exists() and sha(f) == r["pdf_sha256"]) and r["arxiv_version"]:
            req = urllib.request.Request(f"https://arxiv.org/pdf/{r['arxiv_version']}",
                                         headers={"User-Agent": "section-structure-service benchmark fetch"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                f.write_bytes(resp.read())
            time.sleep(3)
        if not f.exists():
            manual.append(f"{r['paper_id']}: https://doi.org/{r['doi']}")
        elif sha(f) == r["pdf_sha256"]:
            ok += 1
        else:
            bad.append(r["paper_id"])
    print(json.dumps({"pdfs_ok": ok, "hash_mismatch": bad, "download_by_hand": manual}, indent=1))


if __name__ == "__main__":
    main()
