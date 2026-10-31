"""Standard-library CLI for paper_structure."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .api import UnknownSectionError, build_document, recover_structure
from .predictions import load_predicted_headings


def _add_document_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("file", type=Path, help="the document: a PDF or a UTF-8 text file")
    parser.add_argument("--json", action="store_true",
                        help="print JSON for programs and LLM agents (PDFs accepted; uses the structure cache)")
    parser.add_argument("--no-cache", action="store_true", help="with --json or a PDF: ignore the structure cache")
    parser.add_argument(
        "--model",
        help="local checkpoint path (or set PAPER_STRUCTURE_MODEL)",
    )
    parser.add_argument(
        "--headings-jsonl",
        type=Path,
        help="saved per-chunk predictions; bypasses model inference",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="inference device (default PAPER_STRUCTURE_DEVICE, else cuda = the benchmark path; "
        "cpu is slower and may differ slightly)",
    )
    parser.add_argument(
        "--toc-skip-frac",
        type=float,
        default=0.05,
        help="front-of-document fraction to deprioritize; use 0 to disable",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m paper_structure.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    outline_parser = subparsers.add_parser("outline", help="print detected outline")
    _add_document_arguments(outline_parser)
    outline_parser.add_argument(
        "--flat", action="store_true", help="do not indent numbered depths"
    )
    outline_parser.add_argument(
        "--mark-empty",
        action="store_true",
        help="append ' [empty]' to sections with no direct body text",
    )
    outline_parser.add_argument("--max-depth", type=int, help="with --json: only sections up to this depth")

    read_parser = subparsers.add_parser("read", help="read one exact section slice")
    _add_document_arguments(read_parser)
    read_parser.add_argument("--section", required=True, help="section id from outline")
    read_parser.add_argument("--offset", type=int, default=0, help="with --json: character offset (paging)")
    read_parser.add_argument("--max-chars", type=int, default=20000, help="with --json: page size in characters")
    read_parser.add_argument(
        "--include-descendants",
        action="store_true",
        help="extend through all nested descendant sections",
    )

    mcp_parser = subparsers.add_parser("mcp", help="run the MCP server on stdio (for Claude Code, Cursor, Codex ...)")
    mcp_parser.add_argument("--model", help="local checkpoint path (or PAPER_STRUCTURE_MODEL / config file)")
    mcp_parser.add_argument("--device", choices=["cuda", "cpu"])
    tools_parser = subparsers.add_parser("tools", help="print the agent tool definitions as JSON")
    tools_parser.add_argument("--format", choices=["mcp", "openai", "anthropic", "openapi"], default="mcp")
    tools_parser.add_argument("--server-url", default="http://127.0.0.1:8765", help="with --format openapi")
    app_parser = subparsers.add_parser(
        "app", help="local web viewer + the tools over HTTP (POST /v1/tools/<name>, GET /openapi.json); 127.0.0.1 only")
    app_parser.add_argument("--model", help="local checkpoint path (or set PAPER_STRUCTURE_MODEL)")
    app_parser.add_argument("--device", choices=["cuda", "cpu"],
                            help="inference device (default PAPER_STRUCTURE_DEVICE, else cuda)")
    app_parser.add_argument("--port", type=int, default=8765)
    app_parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    app_parser.add_argument("--cache-dir", type=Path, default=None,
                            help="where predicted headings are cached (default ~/.paper_structure/cache)")
    app_parser.add_argument("--no-cache", action="store_true", help="do not read or write the cache")

    search_parser = subparsers.add_parser("search", help="keyword-search one section")
    _add_document_arguments(search_parser)
    search_parser.add_argument("--section", help="section id from outline (default: the whole document)")
    search_parser.add_argument("-q", "--query", required=True, help="literal query")
    search_parser.add_argument("--context-chars", type=int, default=80)
    search_parser.add_argument("--max-results", type=int)
    search_parser.add_argument(
        "--include-descendants",
        action="store_true",
        help="search the section and all nested descendants",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "mcp":
        from .mcp_stdio import main as mcp_main
        return mcp_main(model=args.model, device=args.device)
    if args.command == "tools":
        from . import agent_tools
        tools = {"mcp": agent_tools.mcp_tools, "openai": agent_tools.openai_tools,
                 "anthropic": agent_tools.anthropic_tools,
                 "openapi": lambda: agent_tools.openapi_spec(args.server_url)}[args.format]()
        _write_json(tools)
        return 0
    if args.command == "app":
        from .app import DEFAULT_CACHE, serve
        cache = None if args.no_cache else (args.cache_dir or DEFAULT_CACHE)
        try:
            return serve(model=args.model, device=args.device, port=args.port, open_browser=not args.no_browser,
                         cache_dir=cache)
        except (OSError, ValueError) as exc:   # e.g. the port is in use
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if args.json or (args.headings_jsonl is None and (_is_pdf(args.file) or
                                                       (args.command == "search" and args.section is None))):
        return _run_navigator(args)
    try:
        return _run(args)
    except UnknownSectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    # OSError: missing paper file, or a model path transformers cannot load; ImportError: torch/transformers absent
    except (OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _write_json(obj) -> None:
    """UTF-8 bytes on stdout whatever the console code page (an agent reads them through a pipe)."""
    sys.stdout.buffer.write((json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    sys.stdout.flush()


def _is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(5) == b"%PDF-"
    except OSError:
        return False


def _run_navigator(args: argparse.Namespace) -> int:
    """--json (and any PDF): the cached Navigator path the agent tools use; errors as JSON with --json."""
    from .service import Navigator, ToolError
    if args.headings_jsonl is not None:
        print("error: --headings-jsonl cannot be combined with --json or a PDF", file=sys.stderr)
        return 2
    nav = Navigator(model=args.model, device=args.device, cache_dir=None if args.no_cache else "default")
    if not args.json:   # a PDF in the plain-text mode: same printing as before
        nav.on_progress = lambda done, total: print(f"\rreading chunk {done}/{total}", end="", file=sys.stderr)
    try:
        if args.command == "outline":
            result = nav.outline(args.file, args.max_depth)
        elif args.command == "read":
            result = nav.read(args.file, args.section, args.include_descendants, args.max_chars, args.offset)
        else:
            result = nav.search(args.file, args.query, args.section, args.include_descendants or args.section is None,
                                args.max_results or 20, args.context_chars)
    except ToolError as exc:
        if args.json:
            _write_json(exc.to_dict())
        else:
            print(f"\nerror: {exc.message}", file=sys.stderr)
        return 2
    if args.json:
        _write_json(result)
        return 0
    print("", file=sys.stderr) if nav.on_progress else None
    if args.command == "outline":
        od = nav.open(args.file)
        lines = od.document.outline(indented=not args.flat)
        if args.mark_empty:
            lines = [ln + (" [empty]" if s.has_empty_body else "") for ln, s in zip(lines, od.document.sections)]
        print("\n".join(lines))
    elif args.command == "read":
        print(result["text"], end="")
    else:
        _write_json(result)
    return 0


def _read_paper(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not UTF-8 plain text ({exc.reason} at byte {exc.start})") from exc


def _run(args: argparse.Namespace) -> int:
    text = _read_paper(args.file)
    if args.headings_jsonl is not None:
        if args.model is not None:
            print(
                "--headings-jsonl supplied; skipping --model inference.",
                file=sys.stderr,
            )
        if args.device is not None:
            print("--headings-jsonl supplied; --device is not used.", file=sys.stderr)
        document = build_document(
            text,
            load_predicted_headings(args.headings_jsonl),
            toc_skip_frac=args.toc_skip_frac,
        )
    else:
        document = recover_structure(
            text,
            model=args.model,
            toc_skip_frac=args.toc_skip_frac,
            device=args.device,
        )

    if args.command == "outline":
        lines = document.outline(indented=not args.flat)
        if args.mark_empty:
            lines = [
                line + (" [empty]" if section.has_empty_body else "")
                for line, section in zip(lines, document.sections)
            ]
        print("\n".join(lines))
        if document.unlocated:
            print(
                f"\n[unlocated headings: {len(document.unlocated)}]",
                file=sys.stderr,
            )
        return 0
    if args.command == "read":
        print(
            document.read_section(
                args.section,
                include_descendants=args.include_descendants,
            ),
            end="",
        )
        return 0

    hits = document.search_in_section(
        args.section,
        args.query,
        include_descendants=args.include_descendants,
        context_chars=args.context_chars,
        max_results=args.max_results,
    )
    print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
