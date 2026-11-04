"""Local web viewer: open a PDF or a UTF-8 text file, see the recovered outline, read and search sections, and watch
the agent tools (get_outline / read_section / search_document) with what each result costs in tokens.

It is a view on the same layer the agents use (service.Navigator): the same structure cache, so a file opened here is
instant for an agent afterwards and the other way round. It listens on 127.0.0.1 only and makes no network request.
Uploads are kept under <cache>/uploads/<hash>/<name> so the structure can be found again. Documents are processed one
at a time in a background thread, chunk by chunk, so the page can show progress.
A local program that already has the file (the Zotero plugin: an attachment's path) posts the path to /api/open_path
instead of uploading it; the file is read where it is and not copied.

    paper-structure app [--model PATH] [--device cuda|cpu] [--port 8765] [--no-browser] [--cache-dir DIR]
"""

from __future__ import annotations

import hashlib
import json
import queue
import tempfile
import threading
import time
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, quote, unquote, urlparse

from . import __version__
from .agent_tools import call_tool, openapi_spec
from .api import Document, UnknownSectionError
from .chunking import chunk_text
from .service import DEFAULT_CACHE, Navigator, ToolError, approx_tokens, load_config  # noqa: F401 (load_config: API)

STATIC = Path(__file__).with_name("app_static")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_TOOL_BODY = 1024 * 1024
MAX_DRAIN_BYTES = 16 * 1024 * 1024   # a refused body up to this size is read away so the client gets the answer


def read_input(name: str, data: bytes) -> str:
    """Plain text of uploaded bytes: a PDF (by its %PDF- header) as the benchmark extracts it, else UTF-8 text."""
    if data[:5] == b"%PDF-":
        from .pdf import extract_pdf_text
        return extract_pdf_text(data)
    try:
        return data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is neither a PDF nor UTF-8 text ({exc.reason} at byte {exc.start})") from exc


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class Job:
    def __init__(self, job_id: str, name: str, data: bytes, path: Path | None = None):
        self.job_id, self.name, self.data = job_id, name, data
        self.local = path is not None   # opened in place (open_path), not an upload
        self.status, self.done, self.total = "queued", 0, 0
        self.error: str | None = None
        self.cached = False
        self.seconds: float | None = None
        self.path: Path | None = path
        self.text: str | None = None
        self.headings: list[str] | None = None
        self.document: Document | None = None
        self.section_tokens: dict[str, int] = {}
        self.tokens_full = 0


class App:
    """Jobs and the background worker over one Navigator. ``engine_factory`` lets tests use a fake model."""

    def __init__(self, model: str | None = None, device: str | None = None, cache_dir: Path | str | None = "default",
                 engine_factory: Callable | None = None, config: dict | None = None):
        self.nav = Navigator(model=model, device=device, cache_dir=cache_dir, engine_factory=engine_factory,
                             config=config)
        self.model, self.device = self.nav.model, self.nav.device
        self.uploads = (self.nav.cache_dir / "uploads") if self.nav.cache_dir else Path(tempfile.mkdtemp(prefix="ps-"))
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self._queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, name="paper-structure-worker", daemon=True).start()

    # ---- jobs
    def submit(self, name: str, data: bytes) -> Job:
        with self._lock:
            job = Job(f"j{len(self.order) + 1}", name, data)
            self.jobs[job.job_id] = job
            self.order.append(job.job_id)
        self._queue.put(job)
        return job

    def submit_path(self, path: str) -> Job:
        """Process a local file in place. The path must be an existing file; it is read, never written."""
        p = Path(path).expanduser()
        if not p.is_file():
            raise ValueError(f"no such file: {path}")
        p = p.resolve()
        with self._lock:
            job = Job(f"j{len(self.order) + 1}", p.name, b"", path=p)
            self.jobs[job.job_id] = job
            self.order.append(job.job_id)
        self._queue.put(job)
        return job

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                self._process(job)
            except ToolError as exc:
                job.status, job.error = "error", exc.message
            except Exception as exc:  # noqa: BLE001 - shown to the user, the worker goes on
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"

    def _process(self, job: Job) -> None:
        started = time.time()
        job.status = "reading"
        if not job.local:
            folder = self.uploads / hashlib.sha256(job.data).hexdigest()[:16]
            folder.mkdir(parents=True, exist_ok=True)
            job.path = folder / job.name
            if not job.path.exists():
                job.path.write_bytes(job.data)
            job.data = b""

        def progress(done: int, total: int) -> None:
            job.status, job.done, job.total = "detecting", done, total

        od = self.nav.open(job.path, progress=progress)
        job.cached, job.text, job.headings, job.document = od.from_cache, od.text, od.headings, od.document
        job.tokens_full = approx_tokens(od.text)
        job.section_tokens = {s.id: approx_tokens(od.document.read_section(s.id)) for s in od.document.sections}
        job.total = job.total or len(chunk_text(od.text))
        job.done = job.total
        job.seconds = round(time.time() - started, 1)
        job.status = "done"

    # ---- responses
    def info(self) -> dict:
        return {"model": self.model, "device": self.device, "version": __version__, "tokens": "approx"}

    def job_view(self, job: Job) -> dict:
        out = {"job_id": job.job_id, "name": job.name, "status": job.status, "done": job.done, "total": job.total,
               "error": job.error, "cached": job.cached, "seconds": job.seconds}
        if job.status == "done" and job.document is not None:
            structure = job.document.to_dict()
            od = self.nav.open(job.path)
            for s in structure["sections"]:
                s["tokens"] = job.section_tokens.get(s["id"], 0)
                s["page"] = self.nav.page_of(od, s["char_start"])
            unlocated = [{"predicted_title": u["predicted_title"], "reason": u["reason"]} for u in structure["unlocated"]]
            result = self.nav.outline(job.path)   # exactly what an agent's get_outline returns
            out.update(sections=structure["sections"], unlocated=unlocated,
                       path=str(job.path), pages=len(self.nav.page_starts(od) or []) or None,
                       stats={"chars": len(job.text or ""), "sections": len(structure["sections"]),
                              "headings_predicted": len(job.headings or []), "chunks": job.total,
                              "tokens_full": job.tokens_full},
                       tool_call={"name": "get_outline", "arguments": {"file_path": job.name},
                                  "result_tokens": approx_tokens(_compact(result)), "result": result})
        return out

    def ready(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(f"no job {job_id!r}")
        if job.status != "done" or job.document is None:
            raise RuntimeError(f"job {job_id} is {job.status}")
        return job

    def read(self, job: Job, section: str, descendants: bool) -> dict:
        text = job.document.read_section(section, include_descendants=descendants)
        return {"section_id": section, "text": text, "tokens": approx_tokens(text), "tokens_full": job.tokens_full,
                "tool_call": {"name": "read_section", "arguments": {"file_path": job.name, "section_id": section,
                                                                    "include_subsections": descendants}}}

    def search(self, job: Job, section: str, query: str, descendants: bool) -> dict:
        hits = [asdict(h) for h in job.document.search_in_section(section, query, include_descendants=descendants)]
        return {"hits": hits, "tokens": approx_tokens(_compact(hits)),
                "tool_call": {"name": "search_document", "arguments": {"file_path": job.name, "query": query,
                                                                       "section_id": section,
                                                                       "include_subsections": descendants}}}

    def export(self, job: Job, fmt: str) -> tuple[bytes, str, str]:
        stem = Path(job.name).stem or "document"
        if fmt == "json":
            body = json.dumps({"file": job.name, **job.document.to_dict()}, ensure_ascii=False, indent=1)
            return body.encode("utf-8"), "application/json; charset=utf-8", f"{stem}.structure.json"
        if fmt == "md":
            lines = [f"# {job.name}", ""]
            for s in job.document.sections:
                lines.append(f"{'  ' * (s.depth - 1)}- {' '.join(s.title.split())}")
            if job.document.unlocated:
                lines += ["", "Unlocated predictions:", *[f"- {u.predicted_title} ({u.reason})" for u in job.document.unlocated]]
            return ("\n".join(lines) + "\n").encode("utf-8"), "text/markdown; charset=utf-8", f"{stem}.outline.md"
        raise ValueError("format must be md or json")


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"paper-structure/{__version__}"

        def log_message(self, fmt, *args):  # keep the console quiet
            return

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def _refuse(self, code: int, obj, length: int) -> None:
            """Answer without using the body. A body left unread makes Windows reset the connection, and the client
            then sees 'connection reset' instead of this answer, so read it away first (up to a cap) and close."""
            left = length if 0 < length <= MAX_DRAIN_BYTES else 0
            while left > 0:
                chunk = self.rfile.read(min(left, 1 << 16))
                if not chunk:
                    break
                left -= len(chunk)
            self.close_connection = True
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8",
                       {"Connection": "close"})

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            try:
                if not parts or parts == ["index.html"]:
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if parts == ["api", "info"]:
                    return self._json(200, app.info())
                if parts == ["openapi.json"]:
                    host = self.headers.get("Host") or f"127.0.0.1:{self.server.server_address[1]}"
                    return self._json(200, openapi_spec(f"http://{host}"))
                if parts == ["api", "jobs"]:
                    jobs = [app.jobs[i] for i in reversed(app.order)]
                    return self._json(200, {"jobs": [{"job_id": j.job_id, "name": j.name, "status": j.status,
                                                      "cached": j.cached} for j in jobs]})
                if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                    job = app.jobs.get(parts[2])
                    if job is None:
                        return self._json(404, {"error": f"no job {parts[2]!r}"})
                    return self._json(200, app.job_view(job))
                if len(parts) == 4 and parts[:2] == ["api", "jobs"]:
                    job = app.ready(parts[2])
                    action, desc = parts[3], q.get("descendants") in ("1", "true")
                    if action == "text":
                        return self._json(200, {"text": job.text})
                    if action == "read":
                        return self._json(200, app.read(job, q.get("section", ""), desc))
                    if action == "search":
                        return self._json(200, app.search(job, q.get("section", ""), q.get("q", ""), desc))
                    if action == "export":
                        body, ctype, fname = app.export(job, q.get("format", "md"))
                        return self._send(200, body, ctype, {"Content-Disposition":
                                                             f"attachment; filename*=UTF-8''{quote(fname)}"})
                return self._json(404, {"error": f"not found: {url.path}"})
            except UnknownSectionError as exc:
                return self._json(404, {"error": str(exc)})
            except KeyError as exc:
                return self._json(404, {"error": str(exc.args[0]) if exc.args else "not found"})
            except (RuntimeError, ValueError) as exc:
                return self._json(400, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            if path.startswith("/v1/tools/"):   # the agent tools over HTTP (OpenAPI at /openapi.json)
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                if length < 0 or length > MAX_TOOL_BODY:   # tool arguments are small: refuse without parsing
                    return self._refuse(413, {"error": {"code": "bad_argument",
                                                        "message": f"body must be JSON under {MAX_TOOL_BODY} bytes"}},
                                        length)
                try:
                    args = json.loads(self.rfile.read(length).decode("utf-8") or "{}") if length else {}
                except ValueError:
                    return self._json(400, {"error": {"code": "bad_argument", "message": "the body must be JSON"}})
                result = call_tool(app.nav, unquote(path[len("/v1/tools/"):]), args)
                return self._json(400 if "error" in result else 200, result)
            if path == "/api/open_path":
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                if length <= 0 or length > MAX_TOOL_BODY:
                    return self._refuse(400, {"error": "send JSON {\"path\": ...}"}, length)
                try:
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                    job = app.submit_path(str(body["path"]))
                except (ValueError, KeyError, TypeError) as exc:
                    return self._json(400, {"error": str(exc) if isinstance(exc, ValueError) else "send JSON {\"path\": ...}"})
                return self._json(200, {"job_id": job.job_id, "name": job.name, "status": job.status})
            if path != "/api/open":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return self._json(400, {"error": "empty upload"})
            if length > MAX_UPLOAD_BYTES:
                return self._refuse(400, {"error": f"file larger than {MAX_UPLOAD_BYTES // 2 ** 20} MB"}, length)
            data = self.rfile.read(length)
            name = unquote(self.headers.get("X-Filename") or "document")
            name = Path(name.replace("\\", "/")).name or "document"
            job = app.submit(name, data)
            return self._json(200, {"job_id": job.job_id, "name": job.name, "status": job.status})

    return Handler


def make_server(app: App, port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))


def serve(model: str | None = None, device: str | None = None, port: int = 8765, open_browser: bool = True,
          cache_dir: Path | str | None = "default") -> int:
    app = App(model=model, device=device, cache_dir=cache_dir)
    server = make_server(app, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"paper-structure app: {url}  (model: {app.model or 'not set'}; device: {app.device}; Ctrl+C to stop)",
          flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
