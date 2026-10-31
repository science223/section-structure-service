"""Recover and navigate section structure in academic-paper plain text."""

__version__ = "0.1.0"

from .api import Document, SearchHit, Section, UnknownSectionError, build_document, recover_structure
from .chunking import CHUNK_SIZE, chunk_text
from .infer import ChunkPrediction, InferenceEngine, detect_headings, predict_chunks
from .locate import LocatedHeading, UnlocatedHeading, locate_headings


__all__ = [
    "__version__",
    "CHUNK_SIZE",
    "ChunkPrediction",
    "Document",
    "InferenceEngine",
    "LocatedHeading",
    "SearchHit",
    "Section",
    "UnknownSectionError",
    "UnlocatedHeading",
    "build_document",
    "chunk_text",
    "detect_headings",
    "locate_headings",
    "predict_chunks",
    "recover_structure",
]
