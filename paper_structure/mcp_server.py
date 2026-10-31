"""Optional FastMCP server exposing the three navigation tools."""

from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from .api import Document, build_document, recover_structure
from .infer import resolve_device
from .predictions import load_predicted_headings


@lru_cache(maxsize=16)
def _cached_document(
    file_path: str,
    modified_ns: int,
    file_size: int,
    model_path: str | None,
    headings_jsonl_path: str | None,
    headings_modified_ns: int,
    headings_file_size: int,
    toc_skip_frac: float,
    device: str | None = None,
) -> Document:
    del modified_ns, file_size, headings_modified_ns, headings_file_size
    text = Path(file_path).read_text(encoding="utf-8")
    if headings_jsonl_path is not None:
        return build_document(
            text,
            load_predicted_headings(headings_jsonl_path),
            toc_skip_frac=toc_skip_frac,
        )
    return recover_structure(text, model=model_path, toc_skip_frac=toc_skip_frac, device=device)


def _document(
    file_path: str,
    model_path: str | None,
    toc_skip_frac: float,
    headings_jsonl: str | None = None,
) -> Document:
    path = Path(file_path).expanduser().resolve()
    stat = path.stat()
    resolved_model = model_path or os.environ.get("PAPER_STRUCTURE_MODEL")
    if not headings_jsonl:
        resolved_headings = None
        headings_modified_ns = 0
        headings_file_size = 0
    else:
        headings_path = Path(headings_jsonl).expanduser().resolve()
        headings_stat = headings_path.stat()
        resolved_headings = str(headings_path)
        headings_modified_ns = headings_stat.st_mtime_ns
        headings_file_size = headings_stat.st_size
    return _cached_document(
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        resolved_model,
        resolved_headings,
        headings_modified_ns,
        headings_file_size,
        toc_skip_frac,
        # part of the cache key, so changing PAPER_STRUCTURE_DEVICE in a long-running server takes effect;
        # not used when the headings come from a file
        None if resolved_headings is not None else resolve_device(),
    )


def create_server():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "The optional MCP SDK is not installed. Install 'mcp' in the "
            "server environment; the core library and CLI do not require it."
        ) from exc

    server = FastMCP("paper-structure")

    @server.tool()
    def inspect_structure(
        file_path: str,
        model_path: str | None = None,
        toc_skip_frac: float = 0.05,
        headings_jsonl: str | None = None,
    ) -> dict:
        """Return sections (including has_empty_body) and unlocated headings."""
        return _document(
            file_path,
            model_path,
            toc_skip_frac,
            headings_jsonl,
        ).to_dict()

    @server.tool()
    def read_section(
        file_path: str,
        section_id: str,
        model_path: str | None = None,
        toc_skip_frac: float = 0.05,
        headings_jsonl: str | None = None,
        include_descendants: bool = False,
    ) -> str:
        """Return one section as an exact slice of the original plain text."""
        return _document(
            file_path,
            model_path,
            toc_skip_frac,
            headings_jsonl,
        ).read_section(
            section_id,
            include_descendants=include_descendants,
        )

    @server.tool()
    def search_in_section(
        file_path: str,
        section_id: str,
        query: str,
        model_path: str | None = None,
        toc_skip_frac: float = 0.05,
        context_chars: int = 80,
        max_results: int | None = None,
        headings_jsonl: str | None = None,
        include_descendants: bool = False,
    ) -> list[dict]:
        """Run literal keyword search inside a single recovered section."""
        hits = _document(
            file_path,
            model_path,
            toc_skip_frac,
            headings_jsonl,
        ).search_in_section(
            section_id,
            query,
            include_descendants=include_descendants,
            context_chars=context_chars,
            max_results=max_results,
        )
        return [asdict(hit) for hit in hits]

    return server


def main() -> None:
    create_server().run()


if __name__ == "__main__":
    main()
