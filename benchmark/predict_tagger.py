"""Run the released ONNX tagger on the benchmark chunks and write predictions in the format score.py reads.

    python benchmark/predict_tagger.py <model dir> <chunks dir> <predictions dir> [--threads 4]

<model dir> is the unzipped jev_j01b_a_q5b_onnx release asset (model.onnx, tokenizer.json, ...); <chunks dir> is the
output of make_chunks.py. Each chunk is tagged on its own, as in the reported run (CPU, Viterbi decoding). Resumes:
papers whose prediction file already exists are skipped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper_structure.jev import JevOnnxEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("chunks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    eng = JevOnnxEngine(a.model, threads=a.threads)
    t0 = time.time()
    for f in sorted(a.chunks.glob("*.jsonl")):
        out = a.out / f.name
        if out.exists():
            continue
        rows = []
        for line in open(f, encoding="utf-8"):
            c = json.loads(line)
            rows.append({"chunk_id": c["chunk_id"], "predicted_headings": eng.tag_text(c["text"])})
        tmp = out.with_suffix(".part")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(out)
        print(f"{f.stem}: {len(rows)} chunks, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
