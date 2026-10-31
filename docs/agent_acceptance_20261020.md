# Agent acceptance runs, 2026-10-20

Two LLM agents from different model families answered questions about real documents using **only** the three
paper-structure MCP tools (`get_outline`, `read_section`, `search_document`), connected through
`paper-structure mcp`. The structure model was the v13_2 checkpoint (local GPU). Local paths are reduced to file
names below; the raw transcripts stay with the project, not in this repository.

## Run 1: Qwen (qwen3.8-max) in Claude Code, a research paper

- Connection: `--mcp-config` with the stdio server; every built-in file and shell tool denied (Read, Glob, Grep, LS,
  Bash, Edit, Write, WebFetch, WebSearch), so the agent could not read the PDF any other way.
- Document: `A1_Transformer_Attention.pdf` ("Attention Is All You Need"), not in the structure cache before the run.
- Calls (from the session transcript, not the agent's own account): `get_outline` (the model read the paper: 24
  sections, `from_cache` false), then three `read_section` calls in parallel: `6.1`, `3.2.2`, `5.3`.
- Answers: BLEU 28.4 on WMT 2014 En-De (6.1); h = 8 heads with d_k = 64 (3.2.2); Adam with beta1 0.9, beta2 0.98,
  epsilon 1e-9 (5.3). All three checked against the section texts afterwards.
- Size: the tool results were about 1.8k tokens (outline about 0.75k with the then indented JSON, sections about 1.1k)
  against 9.9k tokens for the whole document: about 18%. The agent itself reported 9% because it counted only the
  section text. The whole session took 41 s, including loading the model and reading the paper.

## Run 2: Codex CLI (GPT family), a 194-page RFC

- Connection: `codex exec -s read-only` with the MCP server given by `-c mcp_servers.paper-structure...` overrides
  (the user's Codex config untouched), `tool_timeout_sec = 900` and `default_tools_approval_mode = "approve"`.
  Without the last setting the first attempt failed: `codex exec` rejected the MCP call ("approval policy is never")
  and the agent answered from memory, saying so. That attempt does not count.
- Document: RFC 9110 (HTTP Semantics, 461,574 characters), not in the cache before the run.
- Calls: `get_outline` (the model read the document: 261 sections), then `read_section` for `15.5.17`, `9.2.1`,
  `4.2.2`. No shell command.
- Answers: 416 Range Not Satisfiable (15.5.17); GET, HEAD, OPTIONS and TRACE are safe (9.2.1); https defaults to TCP
  port 443 (4.2.2). All three checked against the section texts.
- Size: outline about 8.0k tokens plus about 1.8k for the three sections, against 115k for the whole document: about 8.5%.

## What the runs led to

- The RFC's table of contents had been taken for 13 sections, which then held the plain ids (`14.1`) and pushed the
  real sections to `14.1#2`; an agent reading `14.1` would have received one contents line. `get_outline` now skips
  contents-page entries (a duplicated title whose first occurrence has no body): 12 skipped, `read_section("14.1")`
  returns the section.
- The outline JSON is now compact and leaves out derivable fields: about 31% (A1) to 35% (RFC 9110) fewer tokens.
- A locator diagnostic printed on stdout, which would mix into the MCP stream; it now goes to stderr.
- The Codex setting `default_tools_approval_mode = "approve"` is in the README.

## Runs 3 and 4: Qwen (qwen3.8-max) in Claude Code, an IEEE survey, before and after a search fix

Written to check the IEEE numbering added later that night (`II.A`, `III.D.2`): `RAG_for_LLMs_A_Survey.txt`
(arXiv 2312.10997, plain text, 27,383 tokens; not a benchmark test paper), same connection as run 1 (only the three
MCP tools; Read, Glob, Grep, LS, Bash, Edit, Write, WebFetch, WebSearch, Agent, Task and TodoWrite denied).
Questions: the three query-expansion methods; the three quality scores and four abilities used to evaluate RAG; the
method that aligns the scoring functions of retriever and generator with KL divergence. Expected answers, read in the
section texts beforehand: Multi-Query, Sub-Query, Chain-of-Verification (`III.C`); context relevance, answer
faithfulness, answer relevance; noise robustness, negative rejection, information integration, counterfactual
robustness (`VI.C`); RA-DIT (`IV.B`).

- Run 3 (54 s, 7 turns): outline, `III.C`, `VI.B`, search "KL divergence", `VI.C`, `III.D.2`. Answers 1 and 2
  right. Answer 3 wrong: REPLUG (`III.D.2`), a related method that also trains with KL divergence. The search had
  returned one hit, in `III.D.2`: in `IV.B` the words are split by a line break ("KL\ndivergence") and search matched
  the query literally. **A tool bug, found by the run.**
- Fix (commit 454052e): search runs on a view of the text where any run of whitespace is one space and hyphenation
  points (U+00AD, U+FFFE, which pypdfium2 leaves inside words) are gone; positions map back to the text.
- Run 4 (43 s, 6 turns), same brief: outline, `III.C`, `VI.B`, search "KL divergence" (now 2 hits: `III.D.2`, `IV.B`),
  `VI.C`. All three right, answer 3 from the search snippet in `IV.B`. Read with read_section: 2,677 of 27,383
  tokens (about 10%), plus the outline.

Raw records: the benchmark repository, `data/tool_agent_acceptance_20261020/qwen_ieee_run1_search_bug/` and
`qwen_ieee_run2_after_fix/`.

## Clients tried but not verified (10-20, about 05:00)

- **Grok CLI 1.0.62** (xAI; the server in the folder's `.cursor/mcp.json`, read through Grok's Cursor compatibility
  layer; the user's `~/.grok` config not changed). `grok inspect` lists the server, and our server answers
  `initialize` and `tools/list` in 0.07 s, but in four headless runs Grok's `search_tool` reported "No MCP tools are
  available in this session", the first `use_tool` call answered "Tool not found", and later calls stopped with
  "permission required ... a headless run cannot ask for approval". `--allow` with `get_outline`,
  `paper_structure__get_outline`, `paper-structure__get_outline`, `mcp__paper_structure__get_outline` and `use_tool`
  did not approve them. Not pursued further: `--always-approve` is only used with the user's explicit go.
- **OpenCode 1.18.35**: it found and connected the server (`opencode mcp list`: connected), but this Windows install
  has no credentials for its subscription model pool; the only providers are pay-per-use API keys, which are the
  user's call. Not run.

Raw records: the benchmark repository, `data/tool_agent_acceptance_20261020/grok_cli_not_verified/` and
`opencode_no_credentials/`.

## Run 5: Qwen in Claude Code, a technical report with a contents page

To check the contents-page and page-header rules with a real agent: `DeepSeek-V3_Technical_Report.txt` (arXiv
2412.19437, 37,657 tokens; not a benchmark test paper). Its contents page lists every section with dot leaders and
page numbers; before 4ffd531 the contents entry for `3.2.3` held the id and the real section was `3.2.3#2`. Same
connection as run 1. Expected answers, read beforehand: the compressed latent vector c^KV and the decoupled RoPE key
k^R (`2.1.1`); recomputing RMSNorm and the MLA up-projections, EMA kept in CPU memory, the embedding and output head
shared with the MTP module (`3.2.3`); 14.8T tokens and an FIM rate of 0.1 (`4.1`).

- 52 s, 6 turns: outline, then `2.1.1`, `3.2.3`, `3.3.3`, `4.1` in parallel. All three right; answer 2 also adds the
  low-precision storage of `3.3.3` (BF16 optimizer moments, E5M6 and FP8 cached activations), which the section says.
- Read with read_section: 2,028 of 37,657 tokens (about 5.4%), plus the outline.

Raw records: the benchmark repository, `data/tool_agent_acceptance_20261020/qwen_deepseek_v3_contents/`.

## Run 6: a clean install, as a new user or an agent would do it (10-20, about 07:30)

The runs above used the development environment. This one starts from an empty virtual environment (Python 3.13,
Windows), installs exactly what the README says, and talks to `paper-structure mcp` through
`scripts/mcp_probe.py`: plain JSON-RPC lines and no SDK, the way any MCP host does.

- Core install (`pip install -e .`, no dependencies, 6 s): the handshake works and lists the three tools. Documents
  already in the structure cache (DeepSeek-V3, the A1 PDF) are answered without torch or pypdfium2. A new document
  returns a JSON tool error naming what to install, not a stack trace.
- **Found: the install hint pointed at PyPI.** The error said `pip install 'paper-structure[infer]'`, but PyPI has an
  unrelated project with that name (0.1.0 / 0.1.1, a layout and OCR tool uploaded 2026-02-11). Run with `-U`, or in
  an environment without this package, that installs someone else's code. Hints now give the command for the
  folder the package was installed from (`pip install -e "<folder>[infer]"`) and say the PyPI name is unrelated.
- **Found: the README install could not run the model on a GPU.** `pip install -e ".[app]"` on Windows brought
  torch 2.14.1+cpu (pip's default torch there is CPU-only) and no accelerate. First call: transformers refused
  `device_map="auto"` without accelerate (the `[infer]` and `[app]` extras did not list it; the development
  environment happened to have it). After `pip install accelerate`, as that error suggests: the weights loaded, then
  the first chunk failed and the agent received a JSON-RPC internal error, `AssertionError: Torch not compiled with
  CUDA enabled`, with no way forward. Now the extras include accelerate, the engine refuses a CUDA request before
  loading anything when torch has no GPU (the message names the CUDA build and `--device cpu`), and a model failure
  during a chunk (CUDA out of memory, for example) comes back as a `model_error` tool result.
- After the fixes, the README path (CUDA torch from download.pytorch.org, then `pip install -e ".[app]"`; pip kept
  torch 2.11.0+cu128 and added transformers 5.20.2 and accelerate 1.15.0) worked the first time: a new document over
  MCP (model loaded and run: 4.4 s), then A1 (11 s) and DeepSeek-V3 (34 s) with an empty cache. The predicted
  headings are identical to the development environment's (24 of 24, 68 of 68), and so are the outlines.
- Not covered: Linux or macOS, a machine without the checkpoint, the CPU path in a clean environment. On Windows a
  very deep environment path fails earlier, inside pip, because of the 260-character path limit (pypdfium2's
  licence files); that is pip's and the system's, not this package's.

Raw records: the benchmark repository, `data/tool_agent_acceptance_20261020/clean_install/`.
