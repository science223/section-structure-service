# Agent acceptance runs on the released build, 2026-11-04

The two runs of 2026-10-20 (`agent_acceptance_20261020.md`) were repeated with the released tagger. Same task sheets,
same three MCP tools (`get_outline`, `read_section`, `search_document`) through `paper-structure mcp`.

- Package: this repository installed in a clean virtual environment with `[lite,pdf]` (no torch).
- Model: `jev_j01b_a_q5b_onnx` (153 MB ONNX), CPU, an empty cache for each agent.

| Agent | Document | Tool calls | Other tools | Answers | Text read |
|---|---|---|---|---|---|
| Codex CLI (`exec -s read-only`) | RFC 9110 (about 190 pages) | 7: outline 1, search 3, read 3 | none | 416 (15.5.17); GET, HEAD, OPTIONS, TRACE (9.2.1); 443 (4.2.2): all correct | about 1,586 of 115,394 tokens |
| Claude Code with Qwen3.8-Max (only the three MCP tools allowed) | "Attention Is All You Need" | 4: outline 1, read 3 | none (built-in tools disabled) | 28.4 BLEU (6.1); 8 heads, d_k = 64 (3.2.2); Adam 0.9 / 0.98 / 1e-9 (5.3): all correct | 905 of 9,901 tokens |

Tool calls were counted from the session transcripts, not from the agents' own accounts. The raw transcripts stay
with the project, not in this repository.
