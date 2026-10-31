"""Test the tests of the agent layer: break service.py / agent_tools.py / mcp_stdio.py / cli.py / tree.py in known ways, one at a
time, in place, and check that the agent-layer tests (test_agent_tools, test_app, test_api) fail each time (and passes on the unbroken code). Files are
backed up and restored; their SHA-256 is checked afterwards.
    python scripts/mutants_agent_tools.py
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "paper_structure"
MUTANTS = [
    ("service.py", "            from_cache = struct is not None and struct.exists()",
     "            from_cache = False", "cache never used (the model runs every time)"),
    ("service.py", "            idx = bisect.bisect_right(starts, a) - 1",
     "            idx = 0", "search hits all attributed to the first section"),
    ("service.py", "TOC_RUN = 3 ", "TOC_RUN = 1 ", "a single short section counts as a contents page"),
    ("service.py", "            if len(hits) < max_results:\n                snippet",
     "            if len(hits) >= max_results:\n                by_section[key] -= 1\n                continue\n"
     "            if True:\n                snippet", "hits per section counts only the returned hits"),
    ("service.py", "        nxt = offset + len(part) if offset + len(part) < len(text) else None",
     "        nxt = None", "paging never offers the next page"),
    ("service.py", "            raise ToolError(\"unknown_section\", str(exc), known_ids=exc.known) from None\n        s = od",
     "            return {}\n        s = od", "unknown section is not an error"),
    ("agent_tools.py", "        if want is int and isinstance(value, bool) or not isinstance(value, want):",
     "        if False:", "argument types not checked"),
    ("mcp_stdio.py", "                                  \"isError\": \"error\" in result})",
     "                                  \"isError\": False})", "MCP errors not flagged"),
    ("mcp_stdio.py", "        if \"id\" not in msg:      # notifications (initialized, cancelled, ...) need no answer\n            return None",
     "        if False:\n            return None", "notifications answered"),
    ("mcp_stdio.py", "        if not isinstance(msg, dict) or not isinstance(msg.get(\"method\"), str):",
     "        if False:", "bad JSON-RPC messages not rejected"),
    ("cli.py", "    sys.stdout.buffer.write((json.dumps(obj, ensure_ascii=False, separators=(\",\", \":\")) + \"\\n\").encode(\"utf-8\"))",
     "    print(json.dumps(obj, ensure_ascii=False))", "CLI JSON through the console encoding"),
    ("locate.py", "            file=sys.stderr,\n", "", "locator diagnostic back on stdout"),
    ("app.py", "if length < 0 or length > MAX_TOOL_BODY:", "if False:", "HTTP tool body size not limited"),
    ("service.py", "        if not remove:\n            return doc, dropped", "        if True:\n            return doc, dropped",
     "contents entries kept"),
    ("service.py", "                line = lines.after(keys[i], run_end) if empty[i] else None",
     "                line = None", "an entry whose heading the model missed is not moved"),
    ("service.py", "        search = _blank(search, blank, stays)", "        search = search",
     "a late contents page is not blanked out"),
    ("service.py", " or secs[i].id in later_ids:", ":", "a contents entry may take a later section's id"),
    ("service.py", "            elif _lists_titles(", "            elif False and _lists_titles(",
     "a contents entry with the rest of the page as its body is not seen"),
    ("service.py", "                        if old != built:", "                        if False:",
     "structures built by older code pile up in the cache"),
    ("service.py", "                if token:   # contents lines go", "                if False:   #",
     "a numbered running header is kept"),
    ("service.py", "                    keep = next((i for i in group if not thin(i, group)), group[0])",
     "                    keep = group[0]", "the first of the same number is kept even when it is a contents line"),
    ("service.py", "(empty[i] and i + 1 in group)", "empty[i]", "an empty parent heading counts as a page header"),
    ("service.py", "        needle = _SOFT_HYPHENS.sub(\"\", \" \".join(query.split()))", "        needle = query",
     "search does not treat a space as any whitespace"),
    ("service.py", "        elif not _SOFT_HYPHENS.match(piece):", "        else:",
     "hyphenation points stay in the search view"),
    ("service.py", "            line = lines.after(k, page[1]) if len(lines.at.get(k, ())) <= MAX_HEADING_LINES else None",
     "            line = None", "a contents line predicted out of order is not moved to its heading"),
    ("service.py", "if len(lines.at.get(k, ())) <= MAX_HEADING_LINES else None", "if True else None",
     "a page header on a contents page is moved into the text"),
    ("service.py", "            if page is None or k in moved:\n", "            if True:\n",
     "headings predicted on a contents page are left for the locator"),
    ("service.py", "                if keep is not None:   # blank it", "                if False:   #",
     "a dropped earlier line can take the kept heading back"),
    ("service.py", "                        elif any(numbers[n][0] and secs[n].depth <= secs[i].depth",
     "                        elif False and any(numbers[n][0] and secs[n].depth <= secs[i].depth",
     "a template heading repeated per example is taken for a page header"),
    ("service.py", "                        elif any(numbers[n][0] and secs[n].depth", "                        elif any(True and secs[n].depth",
     "an unnumbered caption separates a page header from its heading"),
    ("service.py", "or (in_contents_zone and (empty[i] or next_line_is_a_title(i)))", "or in_contents_zone",
     "an early real heading is taken for a contents line"),
    ("service.py", "(empty[i] or next_line_is_a_title(i))", "(empty[i])",
     "a contents line followed by the next contents line is kept"),
    ("service.py", "                if keep is not None:   # blank it", "                if keep is not None and i < keep:   #",
     "a dropped later page header can take the heading"),
    ("service.py", "                        or interrupts(i)\n", "                        or False\n",
     "a page header set into a sentence is kept instead of the heading"),
    ("service.py", "if len(row) >= TOC_RUN and regular(row):", "if len(row) >= TOC_RUN:",
     "same-titled sections in a row are merged whatever their spacing"),
    ("service.py", '_VIEW_PIECES = re.compile("[\\u00ad\\ufffe]\\\\s*|', '_VIEW_PIECES = re.compile("[\\u00ad\\ufffe]|',
     "a hyphenation point keeps the line break after it in the search view"),
    ("service.py", "        if headings is not None and data.get(\"headings_sha256\") != _headings_sha(headings):",
     "        if False:", "the structure cache ignores a replaced headings cache"),
    ("tree.py", "            if _TOP_LEVEL_RE.fullmatch(t):", "            if _TOP_LEVEL_RE.match(t):",
     "back matter recognised by a prefix of the title"),
    ("service.py", "in_contents_zone = s.char_start < early and any(secs[m].char_start >= early for m in group)",
     "in_contents_zone = False", "a contents line in the first 5% is not recognised"),
    ("service.py", "if len(row) >= TOC_RUN and regular(row):", "if False:",
     "a row of same-titled page headers is kept"),
    ("service.py", "            numbers = outline_numbering([s.title for s in secs])",
     "            numbers = [(None, 1) for s in secs]", "repeats judged without section numbers"),
    ("tree.py", "    ieee = len(roman_at) >= 2", "    ieee = False", "IEEE layout not recognised"),
    ("tree.py", "                    out.append((f\"{roman}.{letter}\", 2))", "                    out.append((letter, 2))",
     "IEEE letters not qualified by their section"),
    ("tree.py", "        if value is not None and last < value <= last + 2:", "        if value is not None:",
     "Roman numbers accepted out of order"),
]


def tests_fail() -> bool:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-B", "-m", "unittest", "tests.test_agent_tools", "tests.test_app",
                        "tests.test_api"], cwd=REPO,
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return r.returncode != 0


def main() -> int:
    files = sorted({f for f, *_ in MUTANTS})
    sha = {f: hashlib.sha256((PKG / f).read_bytes()).hexdigest() for f in files}
    missed = 0
    with tempfile.TemporaryDirectory() as bk:
        for f in files:
            shutil.copy2(PKG / f, Path(bk) / f)
        try:
            if tests_fail():
                print("the unbroken code fails its tests: mutant results would mean nothing")
                return 2
            for f, old, new, name in MUTANTS:
                raw = (PKG / f).read_bytes().decode("utf-8")
                text = raw.replace("\r\n", "\n")
                if text.count(old) != 1:
                    raise SystemExit(f"mutant '{name}' no longer applies to {f}")
                (PKG / f).write_bytes(text.replace(old, new).encode("utf-8"))
                caught = tests_fail()
                missed += not caught
                print(("CAUGHT " if caught else "MISSED ") + name)
                shutil.copy2(Path(bk) / f, PKG / f)
        finally:
            for f in files:
                shutil.copy2(Path(bk) / f, PKG / f)
    ok = all(hashlib.sha256((PKG / f).read_bytes()).hexdigest() == sha[f] for f in files)
    print("files restored unchanged:", ok)
    return 0 if ok and not missed else 1


if __name__ == "__main__":
    sys.exit(main())
