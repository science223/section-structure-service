# Plain-text section tools

`paper_structure` turns UTF-8 academic-paper text into a navigable list of
sections. It reuses the paper benchmark's deterministic 500-word chunking,
local v13_2 heading detector, and three-level heading-to-text locator. Section
bodies are exact slices of the input: the package does not rewrite them.

## Use it from an AI agent

An agent asked about a long document (a paper, an RFC, a specification, API documentation) should not paste the
whole file into its context. These tools let it look at the outline first and read only what it needs:

| Tool | What it returns |
|---|---|
| `get_outline(file_path, max_depth?)` | every section's id, title, depth and approximate token count (with subsections where that differs), plus the document's total; contents-page entries and running page headers the model took for headings are skipped (a contents entry is moved to the heading it lists when the model missed that one); IEEE lettered subsections get their section's number (`II.A`) |
| `read_section(file_path, section_id, include_subsections?, offset?, max_chars?)` | the exact text of one section, paged for long sections (`next_offset`) |
| `search_document(file_path, query, section_id?, include_subsections?, max_results?)` | keyword hits, each with the section it is in, and hits per section |

`file_path` may be a PDF (text extracted exactly as the benchmark did) or a UTF-8 text file. The first call on a new
file runs the local model (about 10-60 s for a paper, a few minutes for a 200-page document); the structure is then
cached on disk, so every later call on that file, from any process, answers without loading the model (a CLI call
on the 460,000-character RFC 9110 took about 0.09 s). Errors come back as JSON (`{"error": {"code": "unknown_section", "message": ..., "known_ids": [...]}}`)
instead of stack traces, so the agent can correct itself.

Install once from this folder (`pip install -e ".[app]"`; on Windows with an NVIDIA GPU, install the CUDA build of
torch first, see [Installation](#installation); the PyPI project named `paper-structure` is an unrelated package)
and tell it where the model is: `--model`, `PAPER_STRUCTURE_MODEL`, or a
UTF-8 file `~/.paper_structure/config.json` containing `{"model": "<checkpoint folder>", "device": "cuda"}` (the
file suits model paths with non-ASCII characters and agents started by other programs).

Then connect it the way your agent takes tools:

- **MCP** (Claude Code, Claude Desktop, Cursor, Codex CLI, any MCP client): `paper-structure mcp` is a stdio server
  with no dependency beyond this package. Claude Code: `claude mcp add paper-structure -- paper-structure mcp`.
  Cursor (`.cursor/mcp.json`) or Claude Desktop (`claude_desktop_config.json`):
  `{"mcpServers": {"paper-structure": {"command": "paper-structure", "args": ["mcp"]}}}`.
  Codex CLI (`~/.codex/config.toml`):

  ```toml
  [mcp_servers.paper-structure]
  command = "paper-structure"
  args = ["mcp"]
  tool_timeout_sec = 900                    # the first read of a long document can take minutes
  default_tools_approval_mode = "approve"   # codex exec rejects MCP calls that need approval otherwise
  ```

  The server sends `notifications/progress` while it reads a new document when the client passes a progress token.
  Tested with real agents (docs/agent_acceptance_20261020.md): Claude Code (with a Qwen model) and Codex CLI. The
  Cursor and Claude Desktop entries above are the documented configuration, not tested here. Grok CLI 1.0.62 finds the
  server (`grok inspect`) but in a headless run its `search_tool` lists no MCP tools and its `--allow` rules did not
  approve them; it may need an interactive approval.
- **Shell** (any agent that can run commands): add `--json` and read stdout (UTF-8 whatever the console):
  `paper-structure outline paper.pdf --json`, `paper-structure read paper.pdf --section 3.2 --json`,
  `paper-structure search paper.pdf -q "learning rate" --json` (whole document; `--section` to narrow it).
- **Function calling** (OpenAI, Anthropic, LangChain, your own loop): `paper-structure tools --format openai` (or
  `anthropic`, `mcp`) prints the tool definitions; run a call with

  ```python
  from paper_structure.agent_tools import call_tool
  from paper_structure.service import Navigator
  nav = Navigator()                       # model from the config file or PAPER_STRUCTURE_MODEL
  result = call_tool(nav, name, arguments)   # name and arguments as the model asked
  ```

- **HTTP + OpenAPI** (Dify, Coze, GPT actions, LangChain's OpenAPI toolkit, plain HTTP): `paper-structure app
  --no-browser` serves `POST /v1/tools/<tool name>` (JSON arguments in, JSON result out; 400 with the error object
  for a bad call) and the spec at `GET /openapi.json` (also `paper-structure tools --format openapi`). It listens on
  127.0.0.1 only.

`paper-structure app` opens a local web viewer on the same layer and cache (outline, highlighted text, the tool calls
with their token cost), handy for checking what an agent will see.

## Evidence and scope

The reported numbers come from a 26-paper benchmark: v13_2 has strict F1
**0.824** and GPT-4o has **0.819**; the two are **statistically
indistinguishable** at n=26. v13_2 is significantly better than the strongest
regex baseline (**0.624**).

Using the saved v13_2 predictions, the locator resolves **1,082 of 1,115
predicted headings (97.04%)** across all 26 papers. Some predictions still
cannot be assigned to source-text positions; every such item is explicitly
listed in `Document.unlocated`.

### Empty direct bodies

On those 26 papers, **79/1,082 sections have an empty direct body**. Of these,
37 contain text when read with `include_descendants=True`, 37 are model false
positives on appendix tables or model-card fields, and the remaining 5 are
individual cases. The Python API (`build_document`, `recover_structure`)
deliberately does **not** filter these predictions: filtering would make its
output inconsistent with the published benchmark predictions. The agent tools
(MCP, CLI `--json`, HTTP; `paper_structure.service.Navigator`) do tidy them:
contents-page entries and running page headers the model marked are dropped or
moved to the heading they name (rules, tests and before/after comparisons on 52
documents: `docs/contents_rule_20261020/README.md`). These numbers were checked
again on 10-20 with `scripts/verify_regression.py gateG` / `gateI` (all pass).

Agents can skip a section when `Section.has_empty_body` is true (or use
`Document.empty_sections()` to obtain all such IDs in source order). For an
empty parent section, reading with `include_descendants=True` includes its
nested content. These signals expose the condition without changing section
boundaries, default reads, or the recovered section count.

The hierarchy tree is a **numbering heuristic**. Unnumbered subsections will be
assigned the wrong depth; `Abstract`, `Introduction`, `References`, and every
other unnumbered heading are depth 1. The one exception is the IEEE layout
(at least two Roman-numeral sections in order): lettered subsections belong to
their Roman section (`II.A`, depth 2), `1)` to the letter (`II.A.1`), and an
unnumbered heading inside a Roman section sits one level below the last numbered
one. Use `document.sections` or `document.outline(indented=False)` when a flat
result is safer.

**Downstream gains for RAG, QA, or summarization have never been evaluated.**
This package makes structure addressable; it does not claim that retrieval or
answer quality improves.

## Requirements

- Python 3.10+
- For inference only: a local Transformers-compatible v13_2 checkpoint, plus
  PyTorch, Transformers and Accelerate (the verified path used torch 2.11.0+cu128 and
  transformers 5.5.0 on CUDA; a clean install on 2026-10-20 with transformers 5.20.2 and accelerate 1.15.0
  predicted the same headings on two documents, see `docs/agent_acceptance_20261020.md`).
  Saved-prediction navigation does not need them.
- Optional MCP server only: the Python `mcp` SDK with `FastMCP`

## Installation

Install only the capability set you need:

```console
pip install -e .              # core location/navigation and offline --headings-jsonl
pip install -e ".[infer]"     # add torch/transformers/accelerate for model inference
pip install -e ".[mcp]"       # add the MCP SDK for the server
pip install -e ".[lite]"      # add onnxruntime/tokenizers for the lightweight ONNX tagger (no torch)
```

- Install from this folder only. This distribution (`plain-text-section-tools`) is not on PyPI, and the PyPI project
  called `paper-structure` is an unrelated layout/OCR tool: `pip install paper-structure` installs someone else's
  code. The command and the Python package here are still called `paper-structure` and `paper_structure`.
- On Windows, pip's default torch is a CPU-only build. For the CUDA path, install the CUDA build first, for example
  `pip install torch --index-url https://download.pytorch.org/whl/cu128`, then `pip install -e ".[infer]"`
  (pip keeps that torch). Without it the first model call says so before loading any weights.
- `[infer]` and `[app]` include `accelerate`: transformers loads the CUDA path (`device_map="auto"`) only with it.

No weights are in this repository. Download `jev_j01b_a_q5b_onnx.zip` from this repository's Releases page (see
`MODEL_CARD.md` inside it), unzip it, and pass that folder explicitly or set
`PAPER_STRUCTURE_MODEL`; `models/`, checkpoint extensions, and `.env` files are
ignored. The inference adapter deliberately preserves the benchmark's CUDA,
bf16, one-chunk-at-a-time generation path.

Without an NVIDIA GPU, pass `--device cpu` (CLI), `device="cpu"` (Python), or
set `PAPER_STRUCTURE_DEVICE=cpu` (also read by the MCP server). The CPU path
loads float32 weights and is slower (about 3.5 s per 500-word chunk on a
20-core desktop CPU; a 26-page paper has roughly 10-60 chunks). It is not the
benchmark path, so its output can differ slightly; on the one paper checked
(11 chunks) it was identical to the GPU output.

### Lightweight tagger (no torch)

A second kind of model folder is supported: the Jev-format heading tagger (ModernBERT-base fine-tuned to tag every
token as heading start / inside / outside, decoded with Viterbi, so every heading is copied verbatim from the text).
`paper_structure` recognises it from the folder itself, so the commands above do not change:

- a folder with `model.onnx` + `tokenizer.json` + `config.json` (quantized ONNX; the released build is 153 MB) runs on
  the CPU with
  `pip install -e ".[lite]"` only (onnxruntime + tokenizers + numpy, about 90 MB installed, against about 4.3 GB for
  torch with CUDA + transformers);
- a folder with the original safetensors weights (598 MB) runs through the `[infer]` stack, GPU or CPU.

Measured during development: on 3,916 held-out arXiv chunks with silver labels the 153 MB build gave the same
headings as the full-precision model on 3,909 chunks (F1 0.9913 against 0.9915), at about 1.1 s per chunk on four
CPU threads.

Set `PAPER_STRUCTURE_MODEL` to the checkpoint directory using your shell's
normal environment-variable syntax. The core and saved-prediction tests do not
need that variable:

```console
python -m unittest discover -s tests -v
```

CI runs only the unit tests with core dependencies; model inference and MCP
require local manual verification.

## Python API

```python
from pathlib import Path
from paper_structure import recover_structure

text = Path("<path-to-paper.txt>").read_text(encoding="utf-8")
document = recover_structure(
    text,
    model=Path("<path-to-v13_2-checkpoint>"),
)

print("\n".join(document.outline(indented=True)))
section_text = document.read_section("4.3")
whole_chapter = document.read_section("4", include_descendants=True)
hits = document.search_in_section("4.3", "baseline")
child_ids = document.children("4")
parent_id = document.parent("4.3")
print(document.unlocated)  # every discarded prediction is explicit
```

`Document.sections` is ordered and each `Section` has `id`, `title`, `depth`,
`char_start`, `char_end`, `parent_id`, and `has_empty_body`. Numbered headings
use printed numbers such as `4.3` as IDs. Unnumbered headings use readable slugs
built from their first six words, such as `abstract`; duplicates receive `#2`,
`#3`, and so on. (Before 2026-10-19 the slug used every word of the heading, so
saved IDs of unnumbered headings longer than six words have changed; run
`outline` again to get the current IDs.) `title` is the exact matched source text (it can contain a
line break when a heading spans two lines); `outline()` shows it on one line.
An unknown section ID raises `UnknownSectionError` (a `KeyError` subclass) that
lists the known IDs; the CLI prints it as one line and exits with status 2. `parent()` returns
the nearest earlier shallower section, `children()` returns direct child IDs in
source order, and `empty_sections()` returns empty-body IDs in source order.

By default, `read_section()` includes the heading and ends exactly where the
next located heading starts. Pass `include_descendants=True` to `read_section()`
or `search_in_section()` to extend the range through nested descendants, ending
before the next section at the same or a shallower depth. The default remains
the exact adjacent-heading slice. Preamble before the first located heading is
intentionally outside the section list.

Location follows the benchmark order: exact substring, case/whitespace-tolerant
match, then punctuation-stripped word-window match. Standalone lines are
preferred. The first 5% of input is deprioritized to avoid table-of-contents
duplicates; set `toc_skip_frac=0` for inputs without that front matter.
Candidate positions are assigned in prediction order with a maximum-retention,
strictly increasing sequence. Predictions with no text match, or candidates
dropped to preserve that sequence, appear in `Document.unlocated` with distinct
reasons.

For already available predictions (tests, audits, or offline demonstrations),
`build_document(text, headings)` performs location and navigation without
loading a model.

## CLI

The three commands recover structure from the supplied file and use the local
checkpoint path from `--model` or `PAPER_STRUCTURE_MODEL`:

```console
python -m paper_structure.cli outline <path-to-paper.txt>
python -m paper_structure.cli read <path-to-paper.txt> --section 4.3
python -m paper_structure.cli search <path-to-paper.txt> --section 4.3 -q baseline
```

Every command also accepts `--headings-jsonl predictions.jsonl`. This flattens
each row's `predicted_headings` list in file order and calls `build_document()`;
it does not load a model or run inference. If both `--headings-jsonl` and
`--model` are supplied, the saved predictions take precedence and the CLI says
on stderr that inference was skipped.

```console
python -m paper_structure.cli outline <path-to-paper.txt> --headings-jsonl <path-to-predictions.jsonl>
python -m paper_structure.cli read <path-to-paper.txt> --section 4 --include-descendants --headings-jsonl <path-to-predictions.jsonl>
python -m paper_structure.cli search <path-to-paper.txt> --section 4 -q baseline --include-descendants --headings-jsonl <path-to-predictions.jsonl>
```

Add `--toc-skip-frac 0` to disable front-matter deprioritization. `outline
--flat` suppresses indentation, while `outline --mark-empty` appends ` [empty]`
to empty-body sections. The marker is opt-in; default outline output is
unchanged. Search is case-insensitive literal keyword matching with local
snippets; it is not vector retrieval.

## Desktop app (local web page)

```console
pip install -e ".[app]"       # torch, transformers and pypdfium2 (PDF text)
paper-structure app --model <checkpoint-dir>
```

The command starts a small server on `127.0.0.1` (port 8765; `--port` to change it) and opens the browser. Drop a
PDF or a UTF-8 text file on the page: it shows the progress chunk by chunk, then the recovered outline, the text with
the headings highlighted, section reading and keyword search, and an *agent view* with the three tool calls an LLM
agent would make (`get_outline`, `read_section`, `search_document`) and what each result costs in tokens
next to the whole document. The outline can be exported as Markdown or JSON.

- Nothing leaves the machine: the server binds to `127.0.0.1` only and makes no network request.
- PDF text is extracted exactly as the benchmark extracted its papers (pypdfium2, page by page); on the 26 benchmark
  PDFs the app's text is byte-identical to the benchmark's texts.
- Predicted headings are cached under `~/.paper_structure/cache` by the text's SHA-256, the model path, the model
  files' sizes and times, and the device, so reopening a file does not run the model (`--no-cache` to turn it off,
  `--cache-dir` to move it). The sections built from them are cached too, keyed also by the version of the code
  that builds them: after an update of this package the first call on a file rebuilds them from the cached headings
  (seconds, no model).
- Settings, in order: `--model` / `--device`, then `PAPER_STRUCTURE_MODEL` / `PAPER_STRUCTURE_DEVICE`, then
  `~/.paper_structure/config.json` (`{"model": "...", "device": "cuda"}`), which is handy when the app is started
  by double-clicking a script.
- A scanned PDF without a text layer has no text to read; run OCR first.
- For PDFs, every section also carries the page it starts on (`page` in `get_outline`, `read_section` and each
  search hit); the page starts come from the same page-by-page extraction (26/26 benchmark PDFs: same text, every
  page's own text at its start, `scripts/check_pages.py`).

## Zotero plugin (Zotero 7, preview)

`python scripts/build_zotero_xpi.py` builds `dist/paper-structure-<version>.xpi`; install it in Zotero
(Tools → Plugins → gear menu → Install Plugin From File). Start the local service first:
`paper-structure app --no-browser` (the plugin talks to it at `http://127.0.0.1:8765`; change it under
Settings → Paper Structure, where "测试连接" shows the service's model and device).

Right-click one or more items → **提取章节结构（本地）**: the item's PDF is processed on this computer (the plugin
sends the attachment's path to `POST /api/open_path`; nothing is uploaded) and a child note lists the sections with
their depth, page and approximate tokens; each title opens the PDF at its page (`zotero://open-pdf/...?page=N`).
Running it again replaces that note. Unlike PDF-bookmark outlines, this works for PDFs without bookmarks. Tests without Zotero: `tests/zotero/test_outline_note.mjs` (Node)
and `tests/test_zotero_plugin.py` (the plugin's own code against a real local service and a fake Zotero).

## Older MCP server (needs the `mcp` SDK)

For agents, prefer `paper-structure mcp` (see "Use it from an AI agent" above): it needs no extra package and takes
file paths, PDF included. This older server is kept for clients that pass saved predictions (`headings_jsonl`).

`paper_structure.mcp_server` exposes `inspect_structure`, `read_section`, and
`search_in_section`. It caches recovered documents by file metadata and model
path so an inspect/read/search loop does not repeatedly run inference. The MCP
SDK is optional and was not present in the original verification environment;
install it only in the environment where the server will run.

All three tools accept `headings_jsonl`; when provided, they use the saved
predictions and do not load a model. The cache includes the resolved prediction
path and its mtime/size as well as the source file metadata. Each section from
MCP `inspect_structure` includes `has_empty_body`. MCP `read_section` and
`search_in_section` also accept `include_descendants` (default `false`).

Example client configuration (replace both paths):

```json
{"mcpServers":{"paper-structure":{
  "command":"python",
  "args":["-m","paper_structure.mcp_server"],
  "env":{"PYTHONPATH":"<path-to-plain-text-section-tools>",
         "PAPER_STRUCTURE_MODEL":"<path-to-v13_2-checkpoint>"}
}}}
```

Start it directly with `python -m paper_structure.mcp_server`. If `mcp` is
absent, startup fails with a local dependency message and does not attempt a
download.

## Offline agent-loop demo

```console
python examples/demo_navigate.py
```

The demo uses synthetic text, the public `Document` navigation surface, a
pluggable strategy protocol, and a hard-coded deterministic policy. Its route
is: inspect outline -> read 4.3 -> notice the baseline definition is absent ->
read 4.2 -> read 4.4 -> stop. It calls no network LLM and is a tool-interface
demo, **not an experiment or evidence of downstream benefit**. A second trace
contrasts the direct slice for section 4 with whole-chapter reading, prints its
4.1/4.2 children, and shows that the descendant range stops before section 5.

## Tests and internal regression scripts

The unit suite is self-contained except for one A1 saved-prediction regression.
Set `PAPER_STRUCTURE_BENCHMARK_DIR` to the external benchmark root to run that
case from any checkout location. Without the variable, the test discovers a
single sibling directory containing the required fixtures; if none exists, it
reports a skip instead of assuming a machine-specific directory layout.

The files under `scripts/` are internal regression/audit helpers. They are kept
in the repository for provenance and are not required by the library, CLI, or
MCP server. Their full gates require separately supplied benchmark full text
and saved predictions via command-line arguments. No benchmark data, PDFs,
gold labels, or model weights are included in this package.

## Provenance

The headers of `chunking.py`, `infer.py`, and `locate.py` identify their source
files and copied mechanisms. Copies are one-way snapshots: changes in this
package do not update the benchmark, and later benchmark changes do not update
this package automatically.
