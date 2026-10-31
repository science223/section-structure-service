"""Importable v13_2 heading inference.

Source: paper_benchmark/src/predict.py, read 2026-08-23.
Copied content: SYSTEM_PROMPT, parse_model_output(), prompt assembly, and the
per-chunk greedy generation/decode path. The prompt and generation arguments
are intentionally byte-for-byte identical to that source. The adaptation wraps
the path in a cached, reusable engine and never writes prediction artifacts.
This is a one-way copy; the two files are not kept in bidirectional sync.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Union

from ._install import NOT_ON_PYPI, install_hint
from .chunking import chunk_text


def parse_model_output(raw: str) -> list:
    """从模型原始输出解析出 heading list。NONE / 空 -> []"""
    raw = raw.strip()
    if not raw or raw == "NONE":
        return []
    # 兼容 JSON 包装
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "result" in parsed:
            raw = parsed["result"].strip()
    except Exception:
        pass
    if not raw or raw == "NONE":
        return []
    headings = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("###H###"):
            title = line[len("###H###"):].strip()
            if title:
                headings.append(title)
    return headings


SYSTEM_PROMPT = """你是一个学术论文标题识别器。从给定文本块中找出所有章节标题，按出现顺序列出。

输出格式规则：
- 每行一个标题：###H### 标题原文
- 标题原文必须和文本中完全一致，一个字不差
- 如果没有任何标题，只输出：NONE
- 不要输出任何解释、序号或其他内容

章节标题的判断标准：
- 顶级章节：Abstract、Introduction、Conclusion、References、数字开头如"1. Method"
- 二级章节：如"2.1 Overview"、"3.2 Training"
- 三级章节：如"2.1.1 Encoder"
- 附录标题：如"A. Appendix"、"A.1 Details"
- 不是标题：正文句子、图注、表注、作者信息
- 标题必须独占一行，行末没有分号（;）
- 带括号的编号项（如"(1) xxx"、"(4) Retrieved skills..."）不是标题"""


DEFAULT_MODEL_ENV = "PAPER_STRUCTURE_MODEL"
DEVICE_ENV = "PAPER_STRUCTURE_DEVICE"
DEVICES = ("cuda", "cpu")


@dataclass(frozen=True)
class ChunkPrediction:
    """One chunk's in-memory inference result."""

    chunk_id: int
    raw_output: str
    predicted_headings: list[str]


class InferenceEngine:
    """Load a local Transformers checkpoint once and reuse it across calls."""

    def __init__(self, model_path: str | Path, device: str | None = None):
        device = resolve_device(device)   # None: PAPER_STRUCTURE_DEVICE, else cuda (checked before any loading)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:   # the core package installs without them
            raise ImportError(
                "Model inference needs torch, transformers and accelerate: "
                f"{install_hint('infer', 'torch transformers accelerate')} "
                "(for an NVIDIA GPU, install the CUDA build of torch from https://pytorch.org/get-started/locally/ "
                f"first), or pass --headings-jsonl with saved predictions; {NOT_ON_PYPI}"
            ) from exc
        if device == "cuda" and not torch.cuda.is_available():   # else the weights load and the first chunk fails
            raise ValueError(
                f"device is cuda but this torch ({torch.__version__}) cannot use a GPU: "
                + ("it is a CPU-only build (pip's default torch on Windows is one); install the CUDA build from "
                   "https://pytorch.org/get-started/locally/" if torch.version.cuda is None
                   else "no usable NVIDIA GPU or driver was found")
                + "; or run on the CPU with --device cpu or PAPER_STRUCTURE_DEVICE=cpu"
            )

        self.model_path, self.device = str(model_path), device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        dtype_kw = _dtype_keyword()
        if device == "cuda":   # the benchmark path, unchanged
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, **{dtype_kw: torch.bfloat16}, device_map="auto"
            )
        else:                  # CPU fallback: float32, slower; output may differ slightly from the benchmark path
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **{dtype_kw: torch.float32})
        self.model.eval()
        # The checkpoint's generation_config carries a max_length; with max_new_tokens also set, transformers warns on
        # every chunk. max_new_tokens takes precedence either way, so dropping max_length changes no output.
        self.model.generation_config.max_length = None
        # Sampling settings in the checkpoint's generation_config are unused by greedy decoding (do_sample=False) and
        # only produce a warning; clearing them changes no output.
        for name in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, name, None)

    def predict_chunk(self, chunk: dict) -> ChunkPrediction:
        """Run the benchmark's exact prompt/generation/decode path once."""
        import torch

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{chunk['text']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = outputs[0][inputs.input_ids.shape[-1]:]
        raw = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        preds = parse_model_output(raw)
        return ChunkPrediction(
            chunk_id=chunk["chunk_id"],
            raw_output=raw,
            predicted_headings=preds,
        )

    def iter_predictions(self, text: str) -> Iterator[ChunkPrediction]:
        for chunk in chunk_text(text):
            yield self.predict_chunk(chunk)


def _dtype_keyword() -> str:
    """transformers renamed from_pretrained's torch_dtype to dtype (deprecated warning in 5.x); keep older versions."""
    import transformers

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


@lru_cache(maxsize=None)
def _cached_engine(model_path: str, device: str):
    """A token-classification checkpoint (Jev format, paper_structure.jev) gets the tagger engine; any other checkpoint
    the generative engine above, unchanged."""
    from .jev import JevEngine, JevOnnxEngine, is_jev_checkpoint, is_jev_onnx

    if is_jev_onnx(model_path):          # lightweight export: onnxruntime on the CPU, no torch needed
        return JevOnnxEngine(model_path)
    if is_jev_checkpoint(model_path):
        return JevEngine(model_path, device)
    return InferenceEngine(model_path, device)


def resolve_device(device: str | None = None) -> str:
    device = device or os.environ.get(DEVICE_ENV) or "cuda"
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    return device


ModelArgument = Union[str, Path, InferenceEngine, None]


def get_engine(model: ModelArgument = None, device: str | None = None):
    """Resolve a supplied engine/path or PAPER_STRUCTURE_MODEL and cache loads (per path and device)."""
    from .jev import JevEngine, JevOnnxEngine

    if isinstance(model, (InferenceEngine, JevEngine, JevOnnxEngine)):
        return model
    model_path = str(model) if model is not None else os.environ.get(DEFAULT_MODEL_ENV)
    if not model_path:
        raise ValueError(
            "A local model path is required via model=... or "
            f"the {DEFAULT_MODEL_ENV} environment variable."
        )
    if not Path(model_path).is_dir():   # else transformers takes it for a Hub repo id and reports that instead
        raise FileNotFoundError(f"Model folder not found: {model_path} (a local checkpoint folder is required)")
    return _cached_engine(model_path, resolve_device(device))


def predict_chunks(text: str, model: ModelArgument = None, device: str | None = None) -> list[ChunkPrediction]:
    """Return in-memory predictions for every deterministic 500-word chunk."""
    return list(get_engine(model, device).iter_predictions(text))


def detect_headings(text: str, model: ModelArgument = None, device: str | None = None) -> list[str]:
    """Detect headings in prediction order, returning a single flat list."""
    return [
        heading
        for prediction in predict_chunks(text, model, device)
        for heading in prediction.predicted_headings
    ]
