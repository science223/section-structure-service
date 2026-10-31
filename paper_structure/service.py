"""The layer under every agent interface (MCP server, CLI --json, HTTP API, Python tool functions).

An agent works with file paths. ``Navigator.open(path)`` reads a PDF (text extracted exactly as the benchmark did) or a
UTF-8 text file, recovers the section structure once with the local model, and caches it on disk keyed by the text's
SHA-256, the model path and the device. Every later call on the same file (from any process: each CLI call is a new
process) loads the cached structure and the cached text instead of the model, so reading and searching cost about
the time it takes to read a small JSON file.

Answers are plain JSON-serialisable dicts, compact and self-describing so an LLM can use them directly:
  outline(path)                      -> sections (id, title, depth, parent, approximate tokens) + how to continue
  read(path, section_id, ...)        -> the exact text of one section, paged by max_chars / offset
  search(path, query, section_id?)   -> keyword hits with the section each one is in (whole document by default)
Errors are ``ToolError(code, message)``; ``code`` is one of the ERROR_CODES below.
Token counts are approximate (characters / 4): good enough for an agent deciding what to read.
"""

from __future__ import annotations

import bisect
import functools
import hashlib
import json
import os
import re
import tempfile
import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

from .api import Document, Section, UnknownSectionError, _document_from_locations
from .locate import UnlocatedHeading, locate_headings
from .tree import outline_numbering
from .chunking import chunk_text
from .infer import DEFAULT_MODEL_ENV, get_engine, resolve_device

CACHE_ENV = "PAPER_STRUCTURE_CACHE"
CONFIG = Path.home() / ".paper_structure" / "config.json"
DEFAULT_CACHE = Path.home() / ".paper_structure" / "cache"
ERROR_CODES = ("file_not_found", "unreadable_file", "no_text", "no_model", "model_error", "unknown_section",
               "bad_argument")
READ_MAX_CHARS = 20000


class ToolError(Exception):
    def __init__(self, code: str, message: str, **extra):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra

    def to_dict(self) -> dict:
        return {"error": {"code": self.code, "message": self.message, **self.extra}}


def approx_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def load_config(path: Path | None = None) -> dict:
    """``~/.paper_structure/config.json`` ({"model": ..., "device": ..., "cache_dir": ...}): a UTF-8 file, so a model
    path may hold any characters; missing or unreadable means no settings."""
    try:
        data = json.loads((path or CONFIG).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_file_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise ToolError("file_not_found", f"No such file: {path}") from None
    except OSError as exc:   # a folder, no permission, ...
        raise ToolError("unreadable_file", f"Cannot read {path}: {exc}") from None


def read_document_text(path: Path) -> str:
    """Plain text of a PDF (by its %PDF- header; extracted as the benchmark did) or of a UTF-8 text file."""
    return document_text(path, read_file_bytes(path))


def document_text(path: Path, data: bytes) -> str:
    if data[:5] == b"%PDF-":
        from .pdf import extract_pdf_text
        try:
            return extract_pdf_text(data)
        except ImportError as exc:
            raise ToolError("unreadable_file", str(exc)) from None
        except Exception as exc:  # noqa: BLE001 - a broken PDF
            raise ToolError("unreadable_file", f"{path.name} is not a readable PDF: {exc}") from None
    try:
        return data.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as exc:
        raise ToolError("unreadable_file", f"{path.name} is neither a PDF nor UTF-8 text "
                                           f"({exc.reason} at byte {exc.start})") from None


@dataclass
class OpenDocument:
    path: Path
    sha256: str
    text: str
    headings: list[str]
    document: Document
    from_cache: bool
    toc_dropped: list[str] = field(default_factory=list)
    repeats_dropped: list[str] = field(default_factory=list)
    _ends: dict[str, int] = field(default_factory=dict)
    _pages: list = field(default_factory=list)   # [] not looked up yet; [None] not a PDF; [starts] page start positions

    @property
    def name(self) -> str:
        return self.path.name


_SOFT_HYPHENS = re.compile("[\u00ad\ufffe]")   # PDF text keeps hyphenation points inside words: "abbrevi\ufffeations"
_VIEW_PIECES = re.compile("[\u00ad\ufffe]\\s*|\\s+|[^\\s\u00ad\ufffe]+")   # a hyphenation point takes the line break after it


@functools.lru_cache(maxsize=8)
def _search_view(text: str) -> tuple[str, list[int]]:
    """The text as search sees it, with each character's position in the real text: every run of whitespace (line
    breaks included) is one space and hyphenation points are gone, so "KL divergence" finds "KL\\ndivergence" and
    "abbreviations" finds "abbrevi\\ufffeations". Built once per text (the open document keeps the same string)."""
    out: list[str] = []
    at: list[int] = []
    for m in _VIEW_PIECES.finditer(text):
        piece = m.group()
        if piece[0].isspace():
            out.append(" ")
            at.append(m.start())
        elif not _SOFT_HYPHENS.match(piece):
            out.append(piece)
            at.extend(range(m.start(), m.end()))
    return "".join(out), at


def _title_key(title: str) -> str:
    return " ".join(re.sub(r"[.\s ]+", " ", title).split()).casefold()


TOC_RUN = 3   # consecutive contents-like sections needed before any is dropped
TOC_ZONE = 0.05   # the first 5% of the text, where contents pages are (the locator's toc_skip_frac)
MAX_HEADING_LINES = 3   # a title on more lines of its own than this is a page header, not a heading to move to
PAGE_CHARS = 1000   # page headers of a row repeat at least this far apart (a page holds about 3,000-5,000 characters)


class _Lines:
    """Every non-empty line of the text by its title key, so 'does this title come later as a line of its own' is a
    dictionary lookup instead of a scan of the whole text."""

    def __init__(self, text: str):
        self.at: dict[str, list[tuple[int, str]]] = {}
        pos = 0
        for line in text.split("\n"):
            s = line.strip()
            if s:
                self.at.setdefault(_title_key(s), []).append((pos + len(line) - len(line.lstrip()), s))
            pos += len(line) + 1

    def after(self, key: str, pos: int) -> tuple[int, str] | None:
        """The first line at or after `pos` whose key is `key`: (start, the line's text)."""
        return next(((p, s) for p, s in self.at.get(key, ()) if p >= pos), None)


def _lists_titles(body: str, keys: set[str], lines: _Lines, end: int) -> bool:
    """True when at least half the lines of `body` (and TOC_RUN or more) are titles of later sections in `keys`, or
    title-sized lines that come again after `end` on a line of their own (page numbers after a title allowed)."""
    rows = [ln for ln in body.split("\n") if ln.strip()]
    if len(rows) < TOC_RUN:
        return False
    hits = 0
    for ln in rows:
        k = _title_key(ln)
        for key in {k, re.sub(r" \d+$", "", k)}:
            if key in keys or (8 <= len(key) <= 120 and lines.after(key, end) is not None):
                hits += 1
                break
    return 2 * hits >= len(rows)


def drop_toc_entries(text: str, headings: list[str]) -> tuple[Document, list[str]]:
    """tidy_headings without the running-header rule: the document and the dropped contents entries."""
    doc, dropped, _ = tidy_headings(text, headings, repeats=False)
    return doc, dropped


def tidy_headings(text: str, headings: list[str], repeats: bool = True) -> tuple[Document, list[str], list[str]]:
    """Build the document, then deal with the table-of-contents entries the model took for headings.
    The model sees a contents page chunk by chunk and often marks its lines; left alone, they take the plain section
    ids ("14.1") and leave the real sections "14.1#2", so an agent reading "14.1" would get one contents line.
    A section is part of a contents page when it is the first with its title and either
      - has essentially no body (whitespace, dots, page numbers) and its title comes again later, as another section
        or as a line of its own the model did not mark there, or
      - its body is mostly lines that are titles of later sections, or come again later on a line of their own (the
        rest of the page, which the model left unmarked).
    Only runs of at least TOC_RUN such sections in a row count (a contents page lists many): one short real section
    ("Results\\nOK.") followed later by a same-titled one is kept. In a run, an entry whose title or id belongs to a
    later section is dropped; an empty entry whose title is only a later line is moved to that line (the model marked
    the title, the contents page just came first), so "13.2" starts where 13.2 really starts; anything else stays.
    Once a run is found, headings are located with the page blanked out (except the titles of entries that stay):
    otherwise the locator, which prefers the earliest of equally good matches, would put the remaining heading of
    each dropped entry back on its contents line whenever the page lies past the first toc_skip_frac of the text.
    With `repeats`, running page headers and single contents lines the model marked are dropped too (their text
    joins the section before). Of sections with the same title and the same printed number, the thin ones go (followed
    by dot leaders and a page number, an empty line right above its namesake, in the first TOC_ZONE of the text while
    a namesake is past it, or a list of later titles: "2.1.1 Attention . . . 7" in the contents), and so does a
    repeat with no heading of the same or a higher level since the last one kept ("3.2 Stage 2 ..." at the top of
    every page of 3.2, with only 3.2.x in between). A repeat after a sibling or a higher heading stays: the same
    numbered headings in every example of a template (D2: "4. Step-by-Step Experiment Plan" in each example idea) or
    in a second document in the file (the GPT-4 report and its system card). Of TOC_RUN or more same-titled
    sections in a row ("REFERENCES" on every page of the references) one stays, the first that is not thin. An
    unnumbered title loses only its dotted contents lines. Two "A. Formulation" in different IEEE sections have
    different numbers ("II.A", "III.A") and stay; so do two "Results" in a row.
    Returns the document, the dropped contents entries and the dropped repeats. The predicted headings themselves
    (and the cache) are unchanged."""
    dropped: list[str] = []
    repeated: list[str] = []
    heads = list(headings)
    moved: set[str] = set()
    lines = _Lines(text)
    search = text   # the text headings are located in: contents pages found so far blanked out
    pages: list[tuple[int, int]] = []   # the contents pages found so far
    for _ in range(5):
        located, unlocated = locate_headings(search, heads)
        doc = _document_from_locations(text, located, unlocated)
        secs = doc.sections
        skip = {u.predicted_index for u in unlocated}
        head_of = [j for j in range(len(heads)) if j not in skip]   # sections are the located heads, in order
        keys = [_title_key(s.title) for s in secs]
        empty = [len(re.sub(r"[\s.·…\d]", "", text[s.char_start + len(s.title):s.char_end])) < 20 for s in secs]
        first_seen: set[str] = set()
        like = []   # index of every contents-like first occurrence
        for i, s in enumerate(secs):
            if keys[i] in first_seen:
                continue
            first_seen.add(keys[i])
            if empty[i]:
                if keys[i] in keys[i + 1:] or (keys[i] not in moved and lines.after(keys[i], s.char_end)):
                    like.append(i)
            elif _lists_titles(text[s.char_start + len(s.title):s.char_end], set(keys[i + 1:]), lines, s.char_end):
                like.append(i)
        toc, run = [], []
        for i in like + [None]:
            if run and (i is None or i != run[-1] + 1):
                if len(run) >= TOC_RUN:
                    toc.append(run)
                run = []
            if i is not None:
                run.append(i)
        remove: set[int] = set()
        inserts: dict[int, list[tuple[int, str]]] = {}
        in_toc = {i for r in toc for i in r}
        blank: list[tuple[int, int]] = []
        for r in toc:
            run_end = secs[r[-1]].char_end
            blank.append((secs[r[0]].char_start, run_end))
            for i in r:
                later_ids = {s.id.split("#")[0] for s in secs[i + 1:]}
                if keys[i] in keys[i + 1:] or secs[i].id in later_ids:   # a later section has its title or its id
                    remove.add(head_of[i])
                    dropped.append(heads[head_of[i]])
                    continue
                line = lines.after(keys[i], run_end) if empty[i] else None
                if line is None:
                    continue
                remove.add(head_of[i])
                moved.add(keys[i])
                pos, title = line
                nxt = next((n for n in range(len(secs)) if n not in in_toc and secs[n].char_start > pos), None)
                inserts.setdefault(len(heads) if nxt is None else head_of[nxt], []).append((pos, title))
        # a heading predicted out of order on a contents page is not located at all (RFC 9110: "Authors' Addresses");
        # it goes to its line after the page. If it has none, or its title is on many lines (a page header such as
        # "RFC 9110 HTTP Semantics June 2022" is on every page), it is dropped: left in, with its contents line
        # blanked, the locator could put it on a later line in place of a real heading
        pages += blank
        for u in unlocated:
            k = _title_key(u.predicted_title)
            page = next(((a, b) for a, b in pages if any(a <= p < b for p in u.candidate_positions)), None)
            if page is None or k in moved:
                continue
            line = lines.after(k, page[1]) if len(lines.at.get(k, ())) <= MAX_HEADING_LINES else None
            remove.add(u.predicted_index)
            if line is None:
                dropped.append(u.predicted_title)
                continue
            moved.add(k)
            pos, title = line
            nxt = next((n for n in range(len(secs)) if n not in in_toc and secs[n].char_start > pos), None)
            inserts.setdefault(len(heads) if nxt is None else head_of[nxt], []).append((pos, title))
        if repeats:
            numbers = outline_numbering([s.title for s in secs])
            early = int(len(text) * TOC_ZONE)

            def drop(i: int, keep: int | None = None) -> None:
                remove.add(head_of[i])
                repeated.append(heads[head_of[i]])
                if keep is not None:   # blank it, or the locator puts the kept heading on this line: it prefers the
                    #                    earliest match, and a match past the first 5% over one inside it
                    blank.append((secs[i].char_start, secs[i].char_start + len(secs[i].title)))

            def interrupts(i: int) -> bool:   # a page header set into running text: the line before stops in the
                s = secs[i]                   # middle of a sentence and the line after goes on in lower case
                prev = text[:s.char_start].rstrip("\n").rsplit("\n", 1)[-1].strip()
                after = text[s.char_start + len(s.title):s.char_end].lstrip(" \t").lstrip("\n").split("\n", 1)[0].strip()
                return bool(prev) and prev[-1] not in ".!?:;)\"'" and bool(after) and after[0].islower()

            def regular(row: list[int]) -> bool:   # page headers: about one page apart, every time
                gaps = [secs[b].char_start - secs[a].char_start for a, b in zip(row, row[1:])]
                core = gaps[1:] if len(gaps) >= 3 else gaps   # the first one may start mid-page
                med = sorted(core)[len(core) // 2]
                return med >= PAGE_CHARS and all(abs(g - med) <= 0.35 * med for g in core)

            def dotted(i: int) -> bool:   # the title's line goes on with dot leaders and a page number
                rest = text[secs[i].char_start + len(secs[i].title):secs[i].char_end].split("\n", 1)[0]
                return re.fullmatch(r"[ \t]*(?:[.·…][ \t]*)*\d{1,4}[ \t]*", rest) is not None

            def next_line_is_a_title(i: int) -> bool:   # a contents line is followed by the next contents line
                rest = text[secs[i].char_start + len(secs[i].title):secs[i].char_end].lstrip("\n").split("\n", 1)[0]
                k = _title_key(rest)
                return bool(k) and (k in set(keys[i + 1:]) or (8 <= len(k) <= 120 and lines.after(k, secs[i].char_end)))

            def thin(i: int, group: list[int]) -> bool:
                """A contents line or a page header rather than the heading of the text that follows it."""
                s = secs[i]
                # an early position alone is not enough: the GPT-4 report's own "1 Introduction" is at 0.3% of the
                # text (short abstract, authors at the end) and its system card has another one
                in_contents_zone = s.char_start < early and any(secs[m].char_start >= early for m in group)
                return (dotted(i) or (empty[i] and i + 1 in group)                 # header right above its heading
                        or interrupts(i)
                        or (in_contents_zone and (empty[i] or next_line_is_a_title(i)))
                        or _lists_titles(text[s.char_start + len(s.title):s.char_end], set(keys[i + 1:]), lines,
                                         s.char_end))

            kept = [i for i in range(len(secs)) if head_of[i] not in remove and i not in in_toc]
            groups: dict[tuple[str, str], list[int]] = {}
            for i in kept:
                groups.setdefault((keys[i], numbers[i][0] or ""), []).append(i)
            for (_, token), group in groups.items():
                if len(group) < 2:
                    continue
                if token:   # contents lines go; a repeat inside the section it repeats is a page header
                    keep = next((i for i in group if not thin(i, group)), group[0])
                    last = keep
                    for i in group:
                        if i == keep:
                            continue
                        if i < keep or thin(i, group):
                            drop(i, keep)
                        elif any(numbers[n][0] and secs[n].depth <= secs[i].depth
                                 for n in range(last + 1, i) if head_of[n] not in remove):
                            last = i   # a numbered sibling or higher heading in between: the same heading again (a
                            #            template repeated per example, a second document in the file), not a page
                            #            header (an unnumbered one does not count: often a caption the model marked)
                        else:
                            drop(i, last)
                else:   # unnumbered: only a contents line goes, and only when its heading is there
                    keep = next((i for i in group if not thin(i, group)), None)
                    for i in group if keep is not None else []:
                        if dotted(i):
                            drop(i, keep)
            rows: list[list[int]] = []
            for i in kept:
                if head_of[i] in remove:
                    continue
                if rows and keys[rows[-1][-1]] == keys[i]:
                    rows[-1].append(i)
                else:
                    rows.append([i])
            for row in rows:
                if len(row) >= TOC_RUN and regular(row):   # three "Proof" sections in a row are not page headers
                    keep = next((i for i in row if not thin(i, row)), row[0])
                    for i in row:
                        if i != keep:
                            drop(i, keep)
        if not remove:
            return doc, dropped, repeated
        heads = [h for j in range(len(heads) + 1)
                 for h in [t for _, t in sorted(inserts.get(j, []))] + ([heads[j]] if j < len(heads) and j not in remove else [])]
        stays = [(secs[i].char_start, secs[i].char_start + len(secs[i].title)) for i in in_toc if head_of[i] not in remove]
        search = _blank(search, blank, stays)
    located, unlocated = locate_headings(search, heads)
    return _document_from_locations(text, located, unlocated), dropped, repeated


def _blank(text: str, spans: list[tuple[int, int]], keep: list[tuple[int, int]]) -> str:
    """`text` with every character of `spans` but newlines turned into spaces, except inside `keep`: same length and
    same line breaks, so positions found in it are positions in `text`."""
    chars = list(text)
    for a, b in spans:
        chars[a:b] = [c if c == "\n" else " " for c in text[a:b]]
    for a, b in keep:
        chars[a:b] = text[a:b]
    return "".join(chars)


def _code_version() -> str:
    """Hash of the code that turns predicted headings into sections; part of the structure cache key, so a change
    to locating, the tree or the contents-page rule never serves a structure built by older code."""
    here = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for name in ("locate.py", "tree.py", "api.py", "service.py"):
        try:
            h.update((here / name).read_bytes())
        except OSError:
            h.update(name.encode())
    return h.hexdigest()[:10]


STRUCTURE_VERSION = _code_version()


def _headings_sha(headings: list[str]) -> str:
    return hashlib.sha256(json.dumps(headings, ensure_ascii=False).encode("utf-8")).hexdigest()


def _load_structure(path: Path, text: str, headings: list[str] | None = None):
    """(Document, dropped contents entries, dropped repeats) from a structure cache file, or None if it cannot be
    read or was built from other headings than `headings` (the headings cache was replaced, e.g. deleted and
    predicted again)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if headings is not None and data.get("headings_sha256") != _headings_sha(headings):
            return None
        sections = [Section(**s) for s in data["sections"]]
        unlocated = [UnlocatedHeading(**{**u, "candidate_positions": tuple(u.get("candidate_positions", ()))})
                     for u in data["unlocated"]]
        return (Document(sections=sections, unlocated=unlocated, _text=text), list(data.get("toc_dropped", [])),
                list(data.get("repeats_dropped", [])))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
    os.replace(tmp, path)


class Navigator:
    """Opens documents by path, caches structures, answers outline / read / search. ``engine_factory`` lets tests and
    callers supply their own model object (anything with ``predict_chunk(chunk) -> ChunkPrediction``)."""

    def __init__(self, model: str | None = None, device: str | None = None, cache_dir: Path | None | str = "default",
                 engine_factory: Callable | None = None, config: dict | None = None):
        config = load_config() if config is None else config
        self.model = model or os.environ.get(DEFAULT_MODEL_ENV) or config.get("model") or None
        self.device = resolve_device(device or os.environ.get("PAPER_STRUCTURE_DEVICE") or config.get("device"))
        if cache_dir == "default":
            cache_dir = os.environ.get(CACHE_ENV) or config.get("cache_dir") or DEFAULT_CACHE
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._engine = None
        self._engine_factory = engine_factory
        self._lock = threading.Lock()
        self._open: dict[tuple, OpenDocument] = {}
        self._mkey: str | None = None
        self.on_progress: Callable[[int, int], None] | None = None   # (chunks done, total) while the model reads

    # ---- caching
    # Keys are content hashes, never file names or times: a file rewritten in place (even with the same size and
    # modification time) gets a new key. <cache>/filemap/<sha256 of the file's bytes> holds the sha256 of its text
    # (one small file per document, so processes sharing the cache never rewrite each other's entries); <text sha>.txt
    # holds the text; <text sha>_<model key>.json the predicted headings.
    def _model_key(self) -> str:
        """Model path, device, and the size and modification time of the checkpoint's weight and config files, so a
        checkpoint replaced in place gets new keys. Computed once per Navigator."""
        if self._mkey is None:
            fingerprint = []
            if self.model and Path(self.model).is_dir():
                for f in sorted(Path(self.model).iterdir()):
                    if f.is_file() and f.suffix in (".safetensors", ".bin", ".json", ".model"):
                        st = f.stat()
                        fingerprint.append(f"{f.name}:{st.st_size}:{st.st_mtime_ns}")
            key = f"{self.model}|{self.device}|{';'.join(fingerprint)}"
            self._mkey = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        return self._mkey

    def _engine_get(self):
        if self._engine is None:
            if self._engine_factory is not None:
                self._engine = self._engine_factory()
            else:
                if not self.model:
                    raise ToolError("no_model", f"No model configured: pass --model, set {DEFAULT_MODEL_ENV}, or put "
                                                f"{{\"model\": \"<checkpoint folder>\"}} in {CONFIG}")
                try:
                    self._engine = get_engine(self.model, self.device)
                except (OSError, ValueError, ImportError) as exc:
                    raise ToolError("model_error", str(exc)) from None
        return self._engine

    def open(self, path: str | Path, progress: Callable[[int, int], None] | None = None) -> OpenDocument:
        progress = progress or self.on_progress
        p = Path(path).expanduser().resolve()
        raw = read_file_bytes(p)
        file_sha = hashlib.sha256(raw).hexdigest()
        memo = (file_sha, self._model_key())
        with self._lock:
            if memo in self._open:   # served from memory: the model did not run for this call
                return replace(self._open[memo], path=p, from_cache=True)
            text = sha = None
            fmap = self.cache_dir / "filemap" / file_sha if self.cache_dir else None
            if fmap is not None and fmap.exists():
                sha = fmap.read_text(encoding="utf-8").strip()
                if (self.cache_dir / f"{sha}.txt").exists():
                    text = (self.cache_dir / f"{sha}.txt").read_bytes().decode("utf-8")
            if text is None:
                text = document_text(p, raw)
                sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            del raw
            if not text.strip():
                raise ToolError("no_text", f"{p.name} has no text layer (a scanned PDF needs OCR first)")
            struct = self.cache_dir / f"{sha[:32]}_{self._model_key()}.json" if self.cache_dir else None
            from_cache = struct is not None and struct.exists()
            if from_cache:
                headings = json.loads(struct.read_text(encoding="utf-8"))["headings"]
            else:
                engine = self._engine_get()
                chunks = chunk_text(text)
                headings = []
                for i, chunk in enumerate(chunks):
                    try:
                        headings += engine.predict_chunk(chunk).predicted_headings
                    except RuntimeError as exc:   # e.g. CUDA out of memory: a tool error the agent sees, not a crash
                        hint = (" (out of GPU memory: close other programs using the GPU, or use --device cpu / "
                                "PAPER_STRUCTURE_DEVICE=cpu)" if "out of memory" in str(exc).lower() else "")
                        raise ToolError("model_error", f"The model failed on chunk {i + 1} of {len(chunks)}: "
                                                       f"{type(exc).__name__}: {exc}{hint}") from None
                    if progress:
                        progress(i + 1, len(chunks))
                if struct is not None:
                    _atomic_write(struct, json.dumps({"file": p.name, "model": self.model, "device": self.device,
                                                      "headings": headings}, ensure_ascii=False))
            if self.cache_dir is not None:
                if not (self.cache_dir / f"{sha}.txt").exists():
                    _atomic_write(self.cache_dir / f"{sha}.txt", text)
                if not fmap.exists():
                    _atomic_write(fmap, sha)
            # the located structure is cached too: locating hundreds of headings in a long text takes seconds
            built = (self.cache_dir / f"{sha[:32]}_{self._model_key()}_{STRUCTURE_VERSION}.doc.json"
                     if self.cache_dir else None)
            loaded = _load_structure(built, text, headings) if (built is not None and built.exists()) else None
            if loaded is not None:
                document, toc, reps = loaded
            else:
                document, toc, reps = tidy_headings(text, headings)
                if built is not None:
                    _atomic_write(built, json.dumps({"sections": [asdict(s) for s in document.sections],
                                                     "unlocated": [asdict(u) for u in document.unlocated],
                                                     "toc_dropped": toc, "repeats_dropped": reps,
                                                     "headings_sha256": _headings_sha(headings)}, ensure_ascii=False))
                    for old in built.parent.glob(f"{sha[:32]}_{self._model_key()}_*.doc.json"):
                        if old != built:   # built by older code: never read again
                            try:
                                old.unlink()
                            except OSError:
                                pass
            doc = OpenDocument(p, sha, text, headings, document, from_cache, toc, reps)
            self._open[memo] = doc
            return doc

    # ---- pages
    def page_starts(self, od: OpenDocument) -> list[int] | None:
        """Where each PDF page starts in the document text (None for text files, or when the PDF's page-by-page text
        is not exactly the cached text). Worked out once per document and cached as <text sha>.pages.json; reading
        the pages again does not run the model."""
        if od._pages:
            return od._pages[0]
        starts = None
        cache = self.cache_dir / f"{od.sha256}.pages.json" if self.cache_dir else None
        if cache is not None and cache.exists():
            try:
                starts = [int(x) for x in json.loads(cache.read_text(encoding="utf-8"))]
            except (OSError, ValueError, TypeError):
                starts = None
        if starts is None:
            try:
                raw = read_file_bytes(od.path)
            except ToolError:
                raw = b""
            if raw[:5] == b"%PDF-":
                from .pdf import extract_pdf_pages
                try:
                    text, found = extract_pdf_pages(raw)
                except Exception:  # noqa: BLE001 - page numbers are an extra, never a failure
                    text, found = None, None
                if text == od.text:
                    starts = found
                    if cache is not None:
                        _atomic_write(cache, json.dumps(starts))
        od._pages.append(starts)
        return starts

    def page_of(self, od: OpenDocument, pos: int) -> int | None:
        starts = self.page_starts(od)
        if not starts:
            return None
        from .pdf import page_of
        return page_of(starts, pos)

    # ---- answers
    @staticmethod
    def _section_ends(od: OpenDocument) -> dict[str, int]:
        """End of each section including its subsections (where the next section of the same or a shallower depth
        starts), for the 'with subsections' token count."""
        if not od._ends:
            secs = od.document.sections
            for i, s in enumerate(secs):
                end = len(od.text)
                for t in secs[i + 1:]:
                    if t.depth <= s.depth:
                        end = t.char_start
                        break
                od._ends[s.id] = end
        return od._ends

    def outline(self, path: str | Path, max_depth: int | None = None) -> dict:
        od = self.open(path)
        ends = self._section_ends(od)
        sections = []
        for s in od.document.sections:
            if max_depth is not None and s.depth > max_depth:
                continue
            own, whole = approx_tokens(od.text[s.char_start:s.char_end]), approx_tokens(od.text[s.char_start:ends[s.id]])
            # compact on purpose (an agent pays per token): no parent (order + depth give the tree); the count with
            # subsections only where it differs from the section's own
            page = self.page_of(od, s.char_start)
            sections.append({"id": s.id, "title": " ".join(s.title.split()), "depth": s.depth, "tokens": own,
                             **({"tokens_with_subsections": whole} if whole != own else {}),
                             **({"page": page} if page is not None else {}),
                             **({"empty": True} if s.has_empty_body else {})})
        front = od.document.sections[0].char_start if od.document.sections else len(od.text)
        starts = self.page_starts(od)
        return {"file": od.name, "total_tokens": approx_tokens(od.text), "front_matter_tokens": approx_tokens(od.text[:front]),
                **({"pages": len(starts)} if starts else {}),
                "sections": sections, "unlocated_headings": [u.predicted_title for u in od.document.unlocated],
                **({"contents_entries_skipped": len(od.toc_dropped)} if od.toc_dropped else {}),
                **({"repeated_headings_skipped": len(od.repeats_dropped)} if od.repeats_dropped else {}),
                "from_cache": od.from_cache,
                "next": "read_section(file_path, section_id) returns one section's exact text; "
                        "search_document(file_path, query) finds which sections mention a term."}

    def read(self, path: str | Path, section_id: str, include_subsections: bool = False, max_chars: int = READ_MAX_CHARS,
             offset: int = 0) -> dict:
        if max_chars <= 0 or offset < 0:
            raise ToolError("bad_argument", "max_chars must be > 0 and offset >= 0")
        od = self.open(path)
        try:
            text = od.document.read_section(section_id, include_descendants=include_subsections)
        except UnknownSectionError as exc:
            raise ToolError("unknown_section", str(exc), known_ids=exc.known) from None
        s = od.document.get_section(section_id)
        part = text[offset: offset + max_chars]
        nxt = offset + len(part) if offset + len(part) < len(text) else None
        page = self.page_of(od, s.char_start)
        return {"file": od.name, "section_id": s.id, "title": " ".join(s.title.split()),
                **({"page": page} if page is not None else {}),
                "include_subsections": include_subsections, "text": part, "offset": offset,
                "chars": len(part), "section_chars": len(text), "tokens": approx_tokens(part),
                "next_offset": nxt,
                **({"note": f"Truncated: call again with offset={nxt} for the rest."} if nxt is not None else {})}

    def search(self, path: str | Path, query: str, section_id: str | None = None, include_subsections: bool = True,
               max_results: int = 20, context_chars: int = 80) -> dict:
        if not query or not query.strip():
            raise ToolError("bad_argument", "query must not be empty")
        if max_results <= 0 or context_chars < 0:
            raise ToolError("bad_argument", "max_results must be > 0 and context_chars >= 0")
        od = self.open(path)
        secs = od.document.sections
        if section_id is not None:
            try:
                lo = od.document.get_section(section_id).char_start
            except UnknownSectionError as exc:
                raise ToolError("unknown_section", str(exc), known_ids=exc.known) from None
            hi = self._section_ends(od)[section_id] if include_subsections else od.document.get_section(section_id).char_end
        else:
            lo, hi = 0, len(od.text)
        starts = [s.char_start for s in secs]
        hits: list[dict] = []
        by_section: dict[str, int] = {}   # over every hit, not only the returned ones
        total = 0
        view, at = _search_view(od.text)
        needle = _SOFT_HYPHENS.sub("", " ".join(query.split()))
        if not needle:
            raise ToolError("bad_argument", "query must contain more than hyphenation marks")
        v_lo, v_hi = bisect.bisect_left(at, lo), bisect.bisect_left(at, hi)
        for m in re.finditer(re.escape(needle), view[v_lo:v_hi], flags=re.IGNORECASE):
            total += 1
            a, b = at[v_lo + m.start()], at[v_lo + m.end() - 1] + 1   # back to positions in the text
            # the section a hit is in: the last section starting at or before it (-1: before the first heading)
            idx = bisect.bisect_right(starts, a) - 1
            sec = secs[idx] if idx >= 0 else None
            key = sec.id if sec else "(front matter)"
            by_section[key] = by_section.get(key, 0) + 1
            if len(hits) < max_results:
                snippet = " ".join(od.text[max(0, a - context_chars): b + context_chars].split())
                page = self.page_of(od, a)
                hits.append({"section_id": sec.id if sec else None,
                             "section_title": " ".join(sec.title.split()) if sec else "(front matter)",
                             "char_start": a, **({"page": page} if page is not None else {}), "snippet": snippet})
        return {"file": od.name, "query": query, "scope": section_id or "whole document", "total_hits": total,
                "returned": len(hits), "hits_per_section": by_section, "hits": hits}
