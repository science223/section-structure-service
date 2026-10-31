"""Jev-format heading tagger (paper_benchmark data/jev_format_20261022, read 2026-10-23).

Instead of generating heading strings, a token-classification model (ModernBERT-base fine-tuned, about 150M parameters)
tags every token of a chunk B (heading start) / I (inside a heading) / O; the tags are decoded with Viterbi under
"no I after O" and each B/I run becomes one heading, copied verbatim from the chunk. Copied from the benchmark repo:
the piece-wise tokenisation (jev_lib.token_offsets), the span decoding (jev_decode.spans / finish) and viterbi().
A one-way copy, not kept in sync. On X2 (447 arXiv papers, blind-judged truth) this tagger with Viterbi scored
0.9859 F1 on the validation half against 0.9759 for M1 (the best generative 0.5B model, v13_2 line + RL), and it ran
about 9x faster on the same GPU (batch 16 both).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from .chunking import chunk_text

O, B, I = 0, 1, 2
PIECE = re.compile(r"\d|[^\w\s]|\s*[^\W\d]+|\s+")
NEG = -1e9


def is_jev_checkpoint(model_path: str | Path) -> bool:
    """A Jev checkpoint is a token-classification model (config.json architectures end in ForTokenClassification)."""
    import json

    cfg = Path(model_path) / "config.json"
    try:
        archs = json.loads(cfg.read_text(encoding="utf-8")).get("architectures") or []
    except (OSError, ValueError):
        return False
    return any(a.endswith("ForTokenClassification") for a in archs)


def token_offsets(tok, text: str):
    """Tokens of the text cut into pieces first (every digit and punctuation mark alone, a word with the whitespace
    before it, other whitespace alone), as in training."""
    pieces = [(m.start(), m.group()) for m in PIECE.finditer(text)]
    if not pieces:
        return [], []
    enc = tok([p for _, p in pieces], add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = [], []
    for (s0, _), pi, po in zip(pieces, enc["input_ids"], enc["offset_mapping"]):
        ids += pi
        offs += [(a + s0, b + s0) for a, b in po]
    return ids, offs


def viterbi(lp):
    """lp: [n, 3] log-probs (O, B, I) as a torch tensor. Best tag path with O->I and start->I forbidden."""
    import torch

    n = lp.shape[0]
    if n == 0:
        return []
    trans = torch.zeros(3, 3)
    trans[O, I] = NEG
    score = lp[0].clone()
    score[I] = NEG
    back = []
    for t in range(1, n):
        best, arg = (score.unsqueeze(1) + trans).max(0)
        score = best + lp[t]
        back.append(arg)
    tag = int(score.argmax())
    path = [tag]
    for arg in reversed(back):
        tag = int(arg[tag])
        path.append(tag)
    return path[::-1]


def spans(offsets, tags):
    out, cur = [], None
    for (a, b), t in zip(offsets, tags):
        if b <= a:
            continue
        if t == B or (t == I and cur is None):
            if cur:
                out.append(cur)
            cur = [a, b]
        elif t == I:
            cur[1] = b
        else:
            if cur:
                out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    return out


def headings_from_spans(text: str, sp) -> list[str]:
    res, seen = [], set()
    for s, e in sp:
        h = re.sub(r"\s+", " ", text[s:e]).strip()
        key = " ".join(h.lower().split())
        if h and key not in seen:
            seen.add(key)
            res.append(h)
    return res


def is_jev_onnx(model_path: str | Path) -> bool:
    """A lightweight Jev export: model.onnx next to tokenizer.json and a token-classification config.json."""
    d = Path(model_path)
    return (d / "model.onnx").is_file() and (d / "tokenizer.json").is_file() and is_jev_checkpoint(d)


def viterbi_np(lp):
    """viterbi() for a numpy [n, 3] log-prob array (the ONNX engine needs no torch)."""
    import numpy as np

    n = lp.shape[0]
    if n == 0:
        return []
    trans = np.zeros((3, 3))
    trans[O, I] = NEG
    score = lp[0].astype(np.float64).copy()
    score[I] = NEG
    back = []
    for t in range(1, n):
        cand = score[:, None] + trans
        back.append(cand.argmax(0))
        score = cand.max(0) + lp[t]
    tag = int(score.argmax())
    path = [tag]
    for arg in reversed(back):
        tag = int(arg[tag])
        path.append(tag)
    return path[::-1]


class JevOnnxEngine:
    """The same tagger run with onnxruntime + the `tokenizers` library only (no torch / transformers): CPU, one chunk
    at a time. On X2 the float32 export gave the same headings as the torch model on 200 of 200 test chunks."""

    def __init__(self, model_path: str | Path, threads: int | None = None):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            from ._install import install_hint

            raise ImportError("The lightweight model needs onnxruntime and tokenizers: "
                              + install_hint("lite", "onnxruntime tokenizers numpy")) from exc
        import json

        d = Path(model_path)
        self.model_path, self.device = str(d), "cpu"
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        cfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8")) if (d / "tokenizer_config.json").exists() else {}
        self.cls = self.tokenizer.token_to_id(cfg.get("cls_token", "[CLS]"))
        self.sep = self.tokenizer.token_to_id(cfg.get("sep_token", "[SEP]"))
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(d / "model.onnx"), so, providers=["CPUExecutionProvider"])

    def _offsets(self, text: str):
        pieces = [(m.start(), m.group()) for m in PIECE.finditer(text)]
        if not pieces:
            return [], []
        encs = self.tokenizer.encode_batch([p for _, p in pieces], add_special_tokens=False)
        ids, offs = [], []
        for (s0, _), e in zip(pieces, encs):
            ids += e.ids
            offs += [(a + s0, b + s0) for a, b in e.offsets]
        return ids, offs

    def tag_text(self, text: str) -> list[str]:
        import numpy as np

        ids, offs = self._offsets(text)
        inp = np.array([[self.cls] + ids + [self.sep]], dtype=np.int64)
        logits = self.session.run(["logits"], {"input_ids": inp, "attention_mask": np.ones_like(inp)})[0][0]
        m = logits.max(-1, keepdims=True)
        lp = logits - m - np.log(np.exp(logits - m).sum(-1, keepdims=True))
        return headings_from_spans(text, spans(offs, viterbi_np(lp[1:1 + len(offs)])))

    def predict_chunk(self, chunk: dict):
        from .infer import ChunkPrediction

        hs = self.tag_text(chunk["text"])
        raw = "\n".join(f"###H### {h}" for h in hs) if hs else "NONE"
        return ChunkPrediction(chunk_id=chunk["chunk_id"], raw_output=raw, predicted_headings=hs)

    def iter_predictions(self, text: str) -> Iterator:
        for c in chunk_text(text):
            yield self.predict_chunk(c)


class JevEngine:
    """Load a Jev checkpoint once; tag chunks in length-sorted batches."""

    def __init__(self, model_path: str | Path, device: str, batch: int = 16, max_batch_tokens: int = 16384,
                 max_len: int = 8192):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.model_path, self.device = str(model_path), device
        self.batch, self.max_batch_tokens, self.max_len = batch, max_batch_tokens, max_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path, dtype=dtype).to(device).eval()

    def _encode(self, text: str):
        ids, offs = token_offsets(self.tokenizer, text)
        input_ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
        return None if len(input_ids) > self.max_len else (input_ids, offs)

    def tag_texts(self, texts: list[str]) -> list[list[str]]:
        import torch

        exs = [self._encode(t) for t in texts]
        order = sorted((i for i, e in enumerate(exs) if e is not None), key=lambda i: len(exs[i][0]))
        groups, cur = [], []
        for i in order:
            if cur and (len(cur) >= self.batch or (len(cur) + 1) * len(exs[i][0]) > self.max_batch_tokens):
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        out: list[list[str]] = [[] for _ in texts]
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        for g in groups:
            n = max(len(exs[i][0]) for i in g)
            ids = torch.full((len(g), n), pad, dtype=torch.long)
            att = torch.zeros((len(g), n), dtype=torch.long)
            for k, i in enumerate(g):
                m = len(exs[i][0])
                ids[k, :m] = torch.tensor(exs[i][0])
                att[k, :m] = 1
            with torch.no_grad():
                logits = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device)).logits
            lps = torch.log_softmax(logits.float(), -1).cpu()
            for i, lp in zip(g, lps):
                offs = exs[i][1]
                tags = viterbi(lp[1:1 + len(offs)])
                out[i] = headings_from_spans(texts[i], spans(offs, tags))
        return out

    def iter_predictions(self, text: str) -> Iterator:
        from .infer import ChunkPrediction

        chunks = list(chunk_text(text))
        heads = self.tag_texts([c["text"] for c in chunks])
        for c, hs in zip(chunks, heads):
            raw = "\n".join(f"###H### {h}" for h in hs) if hs else "NONE"
            yield ChunkPrediction(chunk_id=c["chunk_id"], raw_output=raw, predicted_headings=hs)

    def predict_chunk(self, chunk: dict):
        from .infer import ChunkPrediction

        hs = self.tag_texts([chunk["text"]])[0]
        raw = "\n".join(f"###H### {h}" for h in hs) if hs else "NONE"
        return ChunkPrediction(chunk_id=chunk["chunk_id"], raw_output=raw, predicted_headings=hs)
