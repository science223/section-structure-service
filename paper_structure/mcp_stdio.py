"""A Model Context Protocol server over stdio with no dependency beyond this package.

Any MCP client (Claude Code, Claude Desktop, Cursor, Codex CLI, ...) starts it as a subprocess and talks JSON-RPC 2.0,
one message per line, on stdin / stdout. It offers the three tools of agent_tools.py (get_outline, read_section,
search_document). The model is loaded once, on the first document that is not cached yet, and stays loaded for the
life of the server; while it reads a new document the server sends notifications/progress when the client asked for
them (a progressToken in the request's _meta), because a long document can take minutes the first time.
Everything is written to stdout as UTF-8 bytes (a Windows console code page must not touch the protocol); logs go to
stderr.

    paper-structure mcp [--model PATH] [--device cuda|cpu]
"""

from __future__ import annotations

import json
import sys
import traceback

from . import __version__
from .agent_tools import call_tool, mcp_tools
from .service import Navigator

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
INSTRUCTIONS = (
    "paper-structure navigates long documents (papers, RFCs, specifications, API docs; PDF or text) without reading "
    "them whole. Call get_outline(file_path) first to see the sections with their token counts, then "
    "read_section(file_path, section_id) for the parts you need, or search_document(file_path, query) to find which "
    "sections mention a term. The first call on a new file runs a local model (seconds to minutes); later calls on "
    "the same file are instant.")


class Server:
    def __init__(self, nav: Navigator, out=None):
        self.nav = nav
        self.out = out if out is not None else sys.stdout.buffer

    def send(self, msg: dict) -> None:
        self.out.write(json.dumps(msg, ensure_ascii=False).encode("utf-8") + b"\n")
        self.out.flush()

    def handle(self, msg) -> dict | None:
        """One JSON-RPC message in, the response out (None for a notification, i.e. a request without an "id"
        member; "id": null is a request and gets an answer)."""
        if isinstance(msg, dict) and "method" not in msg and ("result" in msg or "error" in msg):
            return None          # a response to nothing we asked: ignore
        if not isinstance(msg, dict) or not isinstance(msg.get("method"), str):
            # a batch array, a bare value, a request without a method (MCP does not use JSON-RPC batches)
            return {"jsonrpc": "2.0", "id": msg.get("id") if isinstance(msg, dict) else None,
                    "error": {"code": -32600, "message": "Invalid Request: expected one JSON-RPC request object"}}
        if "id" not in msg:      # notifications (initialized, cancelled, ...) need no answer
            return None
        mid, method, params = msg["id"], msg["method"], msg.get("params")
        params = {} if params is None else params
        if not isinstance(params, dict):
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32602, "message": "Invalid params: expected an object"}}
        if method == "initialize":
            asked = params.get("protocolVersion")
            return self._ok(mid, {"protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                                  "capabilities": {"tools": {"listChanged": False}},
                                  "serverInfo": {"name": "paper-structure", "version": __version__},
                                  "instructions": INSTRUCTIONS})
        if method == "ping":
            return self._ok(mid, {})
        if method == "tools/list":
            return self._ok(mid, {"tools": mcp_tools()})
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if not isinstance(name, str):
                return {"jsonrpc": "2.0", "id": mid,
                        "error": {"code": -32602, "message": "Invalid params: tools/call needs a tool name"}}
            token = (params.get("_meta") or {}).get("progressToken")
            if token is not None:
                self.nav.on_progress = lambda done, total: self.send(
                    {"jsonrpc": "2.0", "method": "notifications/progress",
                     "params": {"progressToken": token, "progress": done, "total": total,
                                "message": f"reading chunk {done} of {total}"}})
            try:
                result = call_tool(self.nav, name, args)
            finally:
                self.nav.on_progress = None
            text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))   # compact: an agent pays per token
            return self._ok(mid, {"content": [{"type": "text", "text": text}], "structuredContent": result,
                                  "isError": "error" in result})
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Method not found: {method}"}}

    @staticmethod
    def _ok(mid, result) -> dict:
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def serve(self, stream=None) -> int:
        stream = stream if stream is not None else sys.stdin.buffer
        for raw in stream:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}})
                continue
            try:
                reply = self.handle(msg)
            except Exception as exc:  # noqa: BLE001 - one bad request must not end the server
                traceback.print_exc(file=sys.stderr)
                reply = {"jsonrpc": "2.0", "id": msg.get("id") if isinstance(msg, dict) else None,
                         "error": {"code": -32603, "message": f"Internal error: {type(exc).__name__}: {exc}"}}
            if reply is not None:
                self.send(reply)
        return 0


def main(model: str | None = None, device: str | None = None) -> int:
    return Server(Navigator(model=model, device=device)).serve()
