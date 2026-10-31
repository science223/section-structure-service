"""Deterministic 500-word chunking used by the paper benchmark.

Source: paper_benchmark/src/chunk_and_align.py, read 2026-08-23.
Copied content: CHUNK_SIZE and chunk_text(), which split ``\\S+`` word spans
without overlap and preserve exact character offsets/text slices.
This is a one-way copy; the two files are not kept in bidirectional sync.
"""

import re


CHUNK_SIZE = 500


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    word_iter = list(re.finditer(r"\S+", text))
    chunks = []
    cid = 0
    for i in range(0, len(word_iter), chunk_size):
        batch = word_iter[i : i + chunk_size]
        chunks.append({
            "chunk_id": cid,
            "char_start": batch[0].start(),
            "char_end": batch[-1].end(),
            "text": text[batch[0].start():batch[-1].end()],
        })
        cid += 1
    return chunks
