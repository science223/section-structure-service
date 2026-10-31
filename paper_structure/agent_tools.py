"""The three agent tools, defined once and exported for every way an agent can call them.

    TOOLS                      name, description (written for an LLM), JSON Schema of the arguments
    call_tool(nav, name, args) run one tool on a Navigator; returns a JSON-serialisable dict, or {"error": {...}}
    openai_tools()             OpenAI / Chat Completions function-calling format
    anthropic_tools()          Anthropic Messages API tool format
    mcp_tools()                MCP tools/list format

Any agent framework that takes JSON-Schema tools can use these in a few lines, e.g.:

    from paper_structure.agent_tools import TOOLS, call_tool
    from paper_structure.service import Navigator
    nav = Navigator()                                   # model from PAPER_STRUCTURE_MODEL or the config file
    result = call_tool(nav, "get_outline", {"file_path": "paper.pdf"})
"""

from __future__ import annotations

from .service import READ_MAX_CHARS, Navigator, ToolError

_FILE = {"type": "string", "description": "Path to the document: a PDF or a UTF-8 text file (absolute is safest)."}

TOOLS = [
    {
        "name": "get_outline",
        "description": (
            "Get the section outline of a long document (a research paper, an RFC, a specification, API documentation; "
            "PDF or plain text) instead of reading all of it: every section's id, title, depth (the order and depth give the tree) and approximate "
            "token count, plus the document's total. Call this first, then read only the sections you need with "
            "read_section, or find where a term is discussed with search_document. The first call on a new file runs a "
            "small local model (roughly 10-60 seconds for a paper, a few minutes for a 200-page document); every later "
            "call on the same file is instant because the structure is cached."),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": _FILE,
                "max_depth": {"type": "integer", "minimum": 1,
                              "description": "Only sections up to this depth (1 = top-level sections). Default: all."},
            },
            "required": ["file_path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_section",
        "description": (
            "Return the exact text of one section of a document, by the section id from get_outline (for example "
            "\"4.2\", \"abstract\", \"related-work\"). Set include_subsections to true to get the section together with "
            "all of its subsections (useful when a section has no body of its own). Long sections are paged: when the "
            "result has a next_offset, call again with offset set to it."),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": _FILE,
                "section_id": {"type": "string", "description": "A section id exactly as get_outline returned it."},
                "include_subsections": {"type": "boolean", "default": False},
                "offset": {"type": "integer", "minimum": 0, "default": 0,
                           "description": "Character offset into the section, for the next page."},
                "max_chars": {"type": "integer", "minimum": 1, "default": READ_MAX_CHARS,
                              "description": "Largest page to return, in characters."},
            },
            "required": ["file_path", "section_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_document",
        "description": (
            "Case-insensitive literal keyword search in a document; spaces in the query also match line breaks in "
            "the text. Each hit comes with the id and title of the section it is in and a short snippet, plus a "
            "count of hits per section, so you can read just the right section next. Searches the whole document "
            "unless section_id is given."),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": _FILE,
                "query": {"type": "string", "description": "Text to find (not a regular expression)."},
                "section_id": {"type": "string", "description": "Only search this section. Default: whole document."},
                "include_subsections": {"type": "boolean", "default": True,
                                        "description": "With section_id: also search its subsections."},
                "max_results": {"type": "integer", "minimum": 1, "default": 20},
            },
            "required": ["file_path", "query"],
            "additionalProperties": False,
        },
    },
]
_BY_NAME = {t["name"]: t for t in TOOLS}
_TYPES = {"string": str, "integer": int, "boolean": bool}


def _check(name: str, args: dict) -> dict:
    tool = _BY_NAME.get(name)
    if tool is None:
        raise ToolError("bad_argument", f"Unknown tool {name!r}; the tools are {', '.join(_BY_NAME)}")
    if not isinstance(args, dict):
        raise ToolError("bad_argument", "arguments must be a JSON object")
    props = tool["input_schema"]["properties"]
    for key in args:
        if key not in props:
            raise ToolError("bad_argument", f"{name} has no argument {key!r}; it takes {', '.join(props)}")
    for key in tool["input_schema"]["required"]:
        if key not in args:
            raise ToolError("bad_argument", f"{name} needs {key!r}")
    for key, value in args.items():
        want = _TYPES[props[key]["type"]]
        if want is int and isinstance(value, bool) or not isinstance(value, want):
            raise ToolError("bad_argument", f"{key!r} must be a {props[key]['type']}")
    return args


def call_tool(nav: Navigator, name: str, args: dict) -> dict:
    """Run one tool; never raises for a bad call: errors come back as {"error": {"code", "message", ...}}."""
    try:
        a = _check(name, args)
        if name == "get_outline":
            return nav.outline(a["file_path"], a.get("max_depth"))
        if name == "read_section":
            return nav.read(a["file_path"], a["section_id"], a.get("include_subsections", False),
                            a.get("max_chars", READ_MAX_CHARS), a.get("offset", 0))
        return nav.search(a["file_path"], a["query"], a.get("section_id"), a.get("include_subsections", True),
                          a.get("max_results", 20))
    except ToolError as exc:
        return exc.to_dict()


def openai_tools() -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["input_schema"]}} for t in TOOLS]


def anthropic_tools() -> list[dict]:
    return [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in TOOLS]


def mcp_tools() -> list[dict]:
    return [{"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]} for t in TOOLS]


def openapi_spec(server_url: str = "http://127.0.0.1:8765") -> dict:
    """OpenAPI 3.1 for the HTTP form of the tools (``POST /v1/tools/<name>`` on ``paper-structure app``), for platforms
    that import tools from an OpenAPI document (Dify, Coze, GPT actions, LangChain's OpenAPI toolkit, ...)."""
    from . import __version__
    error = {"type": "object", "properties": {"error": {"type": "object", "properties": {
        "code": {"type": "string"}, "message": {"type": "string"}}}}}
    paths = {}
    for t in TOOLS:
        paths[f"/v1/tools/{t['name']}"] = {"post": {
            "operationId": t["name"], "summary": t["description"].split(". ")[0][:120], "description": t["description"],
            "requestBody": {"required": True, "content": {"application/json": {"schema": t["input_schema"]}}},
            "responses": {"200": {"description": "The result", "content": {"application/json": {"schema": {"type": "object"}}}},
                          "400": {"description": "A bad call: {\"error\": {\"code\", \"message\"}}",
                                  "content": {"application/json": {"schema": error}}}}}}
    return {"openapi": "3.1.0",
            "info": {"title": "paper-structure", "version": __version__,
                     "description": "Navigate long documents (papers, RFCs, specifications; PDF or text) by section "
                                    "instead of reading them whole. Runs locally; the first call on a new file runs "
                                    "a small local model, later calls are instant."},
            "servers": [{"url": server_url}], "paths": paths}
