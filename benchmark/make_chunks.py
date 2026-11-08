"""Rebuild the benchmark's 2,072 text chunks from the PDFs and check them against the gold file's hashes.

    python benchmark/make_chunks.py <pdf dir> <out dir> [--gold benchmark/gold_labels_62.jsonl]

Extracts each PDF with this package's extractor (paper_structure.pdf: pypdfium2 get_text_range page by page, pages
joined with a newline, line breaks normalised to \\n), cuts the text at each chunk's char_start/char_end and writes
<out dir>/<paper_id>.jsonl, one line per chunk: chunk_id, char_start, char_end, is_toc, in_main_table, text. These
files contain the papers' text, so they are for local use and are not distributed.

Checked on 11-08-2026: on Linux with pypdfium2 5.8.0 or 5.13.0 all 62 papers reproduce every chunk byte for byte. On
Windows (same versions) 59 do; AKC_2601_01473, AKC_2601_21611 and AKC_2602_03887 differ in spacing and repeated glyphs
inside figure and math-font text. Gold headings are compared as strings, but a differing text shifts the character
positions, so use Linux (or WSL / Docker) to reproduce the reported numbers exactly.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_structure.pdf import extract_pdf_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--gold", type=Path, default=Path(__file__).with_name("gold_labels_62.jsonl"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    papers = chunks = matched = 0
    differ = []
    for line in open(a.gold, encoding="utf-8"):
        r = json.loads(line)
        pdf = a.pdfs / f"{r['paper_id']}.pdf"
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != r["pdf_sha256"]:
            differ.append([r["paper_id"], "PDF hash differs; wrong version?"])
        text = extract_pdf_text(pdf)
        m = 0
        with open(a.out / f"{r['paper_id']}.jsonl", "w", encoding="utf-8", newline="\n") as fo:
            for c in r["chunks"]:
                t = text[c["char_start"]:c["char_end"]]
                m += hashlib.sha256(t.encode("utf-8")).hexdigest() == c["text_sha256"]
                fo.write(json.dumps({k: c[k] for k in ("chunk_id", "char_start", "char_end", "is_toc", "in_main_table")}
                                    | {"text": t}, ensure_ascii=False) + "\n")
        papers += 1
        chunks += len(r["chunks"])
        matched += m
        if m != len(r["chunks"]):
            differ.append([r["paper_id"], f"{m}/{len(r['chunks'])} chunks match", len(text), r["n_chars"]])
    print(json.dumps({"papers": papers, "chunks": chunks, "chunks_matching_hash": matched, "differences": differ},
                     indent=1))


if __name__ == "__main__":
    main()
