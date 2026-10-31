#!/usr/bin/env python3
"""Redact the paper-text excerpts in scripts/gateG_empty_sections.json for a public release.

Each record's context_before / context_after (up to 120 characters of the original paper text around a section
title) becomes {"chars": <length>, "sha256": <hex of the UTF-8 text>}. Every other field (classification,
evidence, positions, gap statistics) is left exactly as it was, so the gate-G results can still be checked; the
excerpts themselves are no longer distributed (the 17 papers have different licences). Someone holding the same
full text can recompute the hash to confirm a record. scripts/verify_regression.py still writes raw excerpts when
re-run locally.

usage: python scripts/redact_public_excerpts.py [--check]   (--check: verify an already redacted file, write nothing)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent / "gateG_empty_sections.json"
KEYS = ("context_before", "context_after")


def main(argv: list[str]) -> int:
    raw = PATH.read_bytes()
    records = json.loads(raw.decode("utf-8"))
    if "--check" in argv:
        bad = [i for i, r in enumerate(records) for k in KEYS if isinstance(r.get(k), str)]
        print(f"{len(records)} records; raw excerpts left: {len(bad)}")
        return 1 if bad else 0
    out, n = [], 0
    for r in records:
        r2 = {}
        for k, v in r.items():
            if k in KEYS and isinstance(v, str):
                r2[k] = {"chars": len(v), "sha256": hashlib.sha256(v.encode("utf-8")).hexdigest()}
                n += 1
            else:
                r2[k] = v
        out.append(r2)
    for a, b in zip(records, out):   # nothing else changed
        assert {k: v for k, v in a.items() if k not in KEYS} == {k: v for k, v in b.items() if k not in KEYS}
    text = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    if b"\r\n" in raw:
        text = text.replace("\n", "\r\n")
    PATH.write_bytes(text.encode("utf-8"))
    print(f"{len(records)} records; {n} excerpts replaced by length + sha256; other fields unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
