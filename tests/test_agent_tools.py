"""The agent interfaces: Navigator (cache), the tool dispatcher and its exports, the stdio MCP server, CLI --json.
No real model: a fake engine fills the cache, then a separate process reads only the cache."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from paper_structure.agent_tools import TOOLS, anthropic_tools, call_tool, mcp_tools, openai_tools
from paper_structure.infer import ChunkPrediction
from paper_structure.service import Navigator, ToolError

REPO = Path(__file__).resolve().parents[1]
TEXT = ("A Study of Things\nAnna Author\n\nAbstract\nWe study things and baselines.\n\n1 Introduction\n"
        + "Things matter. " * 40
        + "\n\n2 Method\n\n2.1 Data\nWe collect data about things.\n\n"
        + "2.2 Model\nA small model with a baseline comparison.\n\n3 Results\nThe baseline loses. Résumé: ünïcode stays.\n")


class FakeEngine:
    def __init__(self):
        self.calls = 0

    def predict_chunk(self, chunk):
        self.calls += 1
        heads = [ln for ln in chunk["text"].split("\n") if re.fullmatch(r"\d+(\.\d+)* [A-Z]\w+|Abstract", ln)]
        return ChunkPrediction(chunk["chunk_id"], "", heads)


def explode():
    raise AssertionError("the model must not be loaded: the structure is cached")


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.cache = self.dir / "cache"
        self.file = self.dir / "paper.txt"
        self.file.write_text(TEXT, encoding="utf-8")
        self.engine = FakeEngine()
        self.nav = Navigator(model="fake-model", device="cpu", cache_dir=self.cache,
                             engine_factory=lambda: self.engine, config={})

    def tearDown(self):
        self._tmp.cleanup()


class TestNavigator(Base):
    def test_outline_then_cache_in_a_new_navigator(self):
        out = self.nav.outline(self.file)
        self.assertEqual([s["id"] for s in out["sections"]], ["abstract", "1", "2", "2.1", "2.2", "3"])
        self.assertFalse(out["from_cache"])
        sec2 = next(s for s in out["sections"] if s["id"] == "2")
        self.assertTrue(sec2.get("empty"))
        self.assertGreater(sec2["tokens_with_subsections"], sec2["tokens"])
        again = Navigator(model="fake-model", device="cpu", cache_dir=self.cache, engine_factory=explode, config={})
        out2 = again.outline(self.file)
        self.assertTrue(out2["from_cache"])
        self.assertEqual(out2["sections"], out["sections"])
        self.assertEqual(again.read(self.file, "2.1")["text"], "2.1 Data\nWe collect data about things.\n\n")

    def test_a_changed_file_is_read_again(self):
        self.nav.outline(self.file)
        self.file.write_text(TEXT.replace("3 Results", "3 Findings"), encoding="utf-8")
        nav = Navigator(model="fake-model", device="cpu", cache_dir=self.cache, engine_factory=lambda: FakeEngine(),
                        config={})
        self.assertIn("3 Findings", [s["title"] for s in nav.outline(self.file)["sections"]])

    def test_read_pages_reassemble_the_section(self):
        whole = self.nav.read(self.file, "1")["text"]
        parts, offset = [], 0
        while offset is not None:
            page = self.nav.read(self.file, "1", max_chars=97, offset=offset)
            parts.append(page["text"])
            offset = page["next_offset"]
        self.assertEqual("".join(parts), whole)
        self.assertGreater(len(parts), 3)
        self.assertIn("note", self.nav.read(self.file, "1", max_chars=97))

    def test_read_with_subsections(self):
        self.assertEqual(self.nav.read(self.file, "2")["text"], "2 Method\n\n")
        both = self.nav.read(self.file, "2", include_subsections=True)["text"]
        self.assertIn("2.1 Data", both)
        self.assertIn("2.2 Model", both)
        self.assertNotIn("3 Results", both)

    def test_search_whole_document_attributes_sections(self):
        r = self.nav.search(self.file, "BASELINE")
        self.assertEqual(r["total_hits"], 3)
        self.assertEqual([h["section_id"] for h in r["hits"]], ["abstract", "2.2", "3"])
        self.assertEqual(r["hits_per_section"], {"abstract": 1, "2.2": 1, "3": 1})
        front = self.nav.search(self.file, "Anna")
        self.assertEqual(front["hits"][0]["section_id"], None)
        self.assertEqual(front["hits"][0]["section_title"], "(front matter)")
        scoped = self.nav.search(self.file, "baseline", section_id="2")
        self.assertEqual([h["section_id"] for h in scoped["hits"]], ["2.2"])
        self.assertEqual(self.nav.search(self.file, "baseline", "2", include_subsections=False)["total_hits"], 0)
        capped = self.nav.search(self.file, "things", max_results=2)
        self.assertEqual((capped["returned"], capped["total_hits"] > 2), (2, True))

    def test_errors(self):
        with self.assertRaises(ToolError) as cm:
            self.nav.read(self.file, "9")
        self.assertEqual(cm.exception.code, "unknown_section")
        self.assertIn("2.1", cm.exception.extra["known_ids"])
        with self.assertRaises(ToolError) as cm:
            self.nav.outline(self.dir / "missing.pdf")
        self.assertEqual(cm.exception.code, "file_not_found")
        bad = self.dir / "latin1.txt"
        bad.write_bytes("caf\xe9".encode("latin-1"))
        with self.assertRaises(ToolError) as cm:
            self.nav.outline(bad)
        self.assertEqual(cm.exception.code, "unreadable_file")
        empty = self.dir / "empty.txt"
        empty.write_text("  \n", encoding="utf-8")
        with self.assertRaises(ToolError) as cm:
            self.nav.outline(empty)
        self.assertEqual(cm.exception.code, "no_text")
        nomodel = Navigator(model=None, device="cpu", cache_dir=None, config={})
        nomodel.model = None
        with self.assertRaises(ToolError) as cm:
            nomodel.outline(self.file)
        self.assertEqual(cm.exception.code, "no_model")


class TestToolCalls(Base):
    def test_dispatch_and_validation(self):
        f = str(self.file)
        self.assertIn("sections", call_tool(self.nav, "get_outline", {"file_path": f}))
        self.assertEqual(call_tool(self.nav, "read_section", {"file_path": f, "section_id": "3"})["section_id"], "3")
        self.assertEqual(call_tool(self.nav, "search_document", {"file_path": f, "query": "data"})["total_hits"], 2)
        cases = [("nope", {}), ("get_outline", {}), ("get_outline", {"file_path": f, "depth": 1}),
                 ("get_outline", {"file_path": f, "max_depth": True}), ("read_section", {"file_path": f, "section_id": 3}),
                 ("search_document", {"file_path": f, "query": ""}), ("read_section", {"file_path": f, "section_id": "x"})]
        for name, args in cases:
            out = call_tool(self.nav, name, args)
            self.assertIn("error", out, (name, args))
            self.assertIn(out["error"]["code"], ("bad_argument", "unknown_section"), (name, args))

    def test_a_model_failure_is_a_tool_error_and_caches_nothing(self):
        import io
        from paper_structure.mcp_stdio import Server

        class OutOfMemory(FakeEngine):
            def predict_chunk(self, chunk):
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        nav = Navigator(model="fake-model", device="cpu", cache_dir=self.cache, engine_factory=OutOfMemory, config={})
        out = call_tool(nav, "get_outline", {"file_path": str(self.file)})
        self.assertEqual(out["error"]["code"], "model_error")
        self.assertIn("chunk 1 of", out["error"]["message"])
        self.assertIn("--device cpu", out["error"]["message"])
        buf = io.BytesIO()   # over MCP: an isError result the model sees, not a JSON-RPC internal error
        Server(nav, buf).serve(io.BytesIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "get_outline", "arguments": {"file_path": str(self.file)}}}).encode("utf-8") + b"\n"))
        reply = json.loads(buf.getvalue().decode("utf-8"))
        self.assertNotIn("error", reply)
        self.assertTrue(reply["result"]["isError"])
        self.assertIn("sections", self.nav.outline(self.file))   # nothing half-done was cached
        self.assertEqual(self.engine.calls, 1)

    def test_exports_carry_the_same_schemas(self):
        names = [t["name"] for t in TOOLS]
        self.assertEqual(names, ["get_outline", "read_section", "search_document"])
        self.assertEqual([t["function"]["name"] for t in openai_tools()], names)
        self.assertEqual([t["input_schema"] for t in anthropic_tools()], [t["input_schema"] for t in TOOLS])
        self.assertEqual([t["inputSchema"] for t in mcp_tools()], [t["input_schema"] for t in TOOLS])
        for t in TOOLS:
            self.assertGreater(len(t["description"]), 80)


class TestProcesses(Base):
    """A real subprocess reads the cache filled here: the MCP server and the CLI never load a model."""

    def env(self):
        e = dict(os.environ, PAPER_STRUCTURE_MODEL="fake-model", PAPER_STRUCTURE_DEVICE="cpu",
                 PAPER_STRUCTURE_CACHE=str(self.cache), PYTHONPATH=str(REPO), PYTHONIOENCODING="ascii")
        return e

    def test_mcp_stdio_session(self):
        self.nav.outline(self.file)   # fills the cache
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "get_outline", "arguments": {"file_path": str(self.file)}}},
                {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                 "params": {"name": "read_section", "arguments": {"file_path": str(self.file), "section_id": "3"}}},
                {"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                 "params": {"name": "read_section", "arguments": {"file_path": str(self.file), "section_id": "99"}}},
                {"jsonrpc": "2.0", "id": 6, "method": "resources/list"},
                {"jsonrpc": "2.0", "id": 7, "method": "ping"}]
        raw = "\n".join(json.dumps(m, ensure_ascii=False) for m in msgs).encode("utf-8") + b"\n{not json\n"
        p = subprocess.run([sys.executable, "-m", "paper_structure.cli", "mcp"], input=raw, capture_output=True,
                           env=self.env(), timeout=120)
        replies = [json.loads(ln) for ln in p.stdout.decode("utf-8").splitlines() if ln.strip()]
        by_id = {r.get("id"): r for r in replies}
        self.assertEqual(by_id[1]["result"]["serverInfo"]["name"], "paper-structure")
        self.assertEqual(by_id[1]["result"]["protocolVersion"], "2025-06-18")
        self.assertEqual([t["name"] for t in by_id[2]["result"]["tools"]], ["get_outline", "read_section", "search_document"])
        outline = by_id[3]["result"]["structuredContent"]
        self.assertTrue(outline["from_cache"])
        self.assertEqual(json.loads(by_id[3]["result"]["content"][0]["text"]), outline)
        self.assertIn("Résumé: ünïcode stays.", by_id[4]["result"]["structuredContent"]["text"])
        self.assertTrue(by_id[5]["result"]["isError"])
        self.assertEqual(by_id[5]["result"]["structuredContent"]["error"]["code"], "unknown_section")
        self.assertEqual(by_id[6]["error"]["code"], -32601)
        self.assertEqual(by_id[7]["result"], {})
        self.assertEqual(by_id[None]["error"]["code"], -32700)
        self.assertEqual(len(replies), 8)   # nothing for the notification

    def test_cli_json(self):
        self.nav.outline(self.file)
        run = lambda *a: subprocess.run([sys.executable, "-m", "paper_structure.cli", *a], capture_output=True,  # noqa: E731
                                        env=self.env(), timeout=120)
        p = run("outline", str(self.file), "--json", "--max-depth", "1")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([s["id"] for s in json.loads(p.stdout.decode("utf-8"))["sections"]], ["abstract", "1", "2", "3"])
        p = run("read", str(self.file), "--section", "3", "--json")
        self.assertIn("ünïcode", json.loads(p.stdout.decode("utf-8"))["text"])   # UTF-8 even with an ASCII console
        p = run("search", str(self.file), "-q", "baseline", "--json")
        self.assertEqual(json.loads(p.stdout.decode("utf-8"))["total_hits"], 3)
        p = run("read", str(self.file), "--section", "nope", "--json")
        self.assertEqual(p.returncode, 2)
        self.assertEqual(json.loads(p.stdout.decode("utf-8"))["error"]["code"], "unknown_section")
        p = run("tools", "--format", "openai")
        self.assertEqual(len(json.loads(p.stdout.decode("utf-8"))), 3)


class TestRobustness(unittest.TestCase):
    def test_contents_page_entries_do_not_take_the_section_ids(self):
        body = "\n\n".join(f"{h}\n" + "Real text of this part. " * 20 for h in ("1 Introduction", "2 Method", "2.1 Data"))
        text = "Title\nContents\n1 Introduction\n2 Method\n2.1 Data\n\n" + body + "\n"
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "doc.txt"
            f.write_text(text, encoding="utf-8")
            nav = Navigator(model="m", device="cpu", cache_dir=None, engine_factory=FakeEngine, config={})
            out = nav.outline(f)
            self.assertEqual([s["id"] for s in out["sections"]], ["1", "2", "2.1"])
            self.assertEqual(out["contents_entries_skipped"], 3)
            self.assertIn("Real text of this part.", nav.read(f, "1")["text"])
            self.assertNotIn("parent", out["sections"][0])
            self.assertNotIn("tokens_with_subsections", out["sections"][2])   # a leaf
            self.assertIn("tokens_with_subsections", out["sections"][1])      # "2" has "2.1"

    def test_contents_entry_whose_heading_the_model_missed_is_moved(self):
        # RFC 9110, 10-20: the model marked the contents line "13.2. Evaluation of Preconditions" but not the heading
        # in the text, so "13.2" was one empty contents line; it has to start at the real heading
        from paper_structure.service import drop_toc_entries
        body = "".join(f"{h}\nReal text of {h}. " + "More of it. " * 20 + "\n\n"
                       for h in ("1. Introduction", "2. Method", "3. Results", "4. Discussion"))
        text = ("Title\n" + "Front matter. " * 40 + "\nContents\n1. Introduction\n2. Method\n3. Results\n"
                "4. Discussion\n\n" + body)
        heads = ["1. Introduction", "2. Method", "3. Results", "1. Introduction", "2. Method", "4. Discussion"]
        doc, dropped = drop_toc_entries(text, heads)
        self.assertEqual([s.id for s in doc.sections], ["1", "2", "3", "4"])
        for s in doc.sections:
            self.assertTrue(text[s.char_start:].startswith(f"{s.title}\nReal text of {s.title}."), s.id)
        self.assertEqual(dropped, ["1. Introduction", "2. Method"])

    def test_a_contents_line_predicted_out_of_order_goes_to_its_heading(self):
        # RFC 9110: "Authors' Addresses" was predicted before the other contents lines of its chunk, so it could not
        # be located at all; once the page is known, it belongs at its line after the page
        from paper_structure.service import tidy_headings
        part = lambda h: f"{h}\nReal text of {h}. " + "More of it. " * 20 + "\n\n"   # noqa: E731
        text = ("Contents\n2 Method\n3 Results\n4 End\nAuthors\n\n" + part("1 Intro") + part("2 Method")
                + part("3 Results") + part("4 End") + "See the Authors section.\nAuthors\nA. Person, Somewhere.\n")
        heads = ["Authors", "2 Method", "3 Results", "4 End", "1 Intro", "2 Method", "3 Results", "4 End"]
        doc, dropped, _ = tidy_headings(text, heads)
        self.assertEqual([s.id for s in doc.sections], ["1", "2", "3", "4", "authors"])
        self.assertEqual(doc.sections[-1].char_start, text.index("Authors\nA. Person"))
        self.assertEqual(doc.unlocated, [])

    def test_a_page_header_on_a_contents_page_is_not_moved(self):
        from paper_structure.service import tidy_headings
        part = lambda h: f"{h}\nReal text of {h}. " + "More of it. " * 20 + "\nJournal of Things 2024\n\n"   # noqa: E731
        text = ("Contents\n2 Method\n3 Results\n4 End\nJournal of Things 2024\n\n" + part("1 Intro") + part("2 Method")
                + part("3 Results") + part("4 End"))
        heads = ["Journal of Things 2024", "2 Method", "3 Results", "4 End", "1 Intro", "2 Method", "3 Results",
                 "4 End"]
        doc, _, _ = tidy_headings(text, heads)
        self.assertEqual([s.id for s in doc.sections], ["1", "2", "3", "4"])

    def test_a_late_contents_page_does_not_pull_headings_back(self):
        # a contents page past the first 5% of the text (an appendix one): once its entries are dropped, the locator
        # must not put the real headings back on its lines (it prefers the earliest of equally good matches)
        from paper_structure.service import drop_toc_entries
        part = lambda h: f"{h}\nReal text of {h}. " + "More of it. " * 20 + "\n\n"   # noqa: E731
        text = (part("1 Introduction") + part("2 Method") + "Appendix contents\nA.1 Data\nA.2 Models\nA.3 Prompts\n\n"
                + part("A.1 Data") + part("A.2 Models") + part("A.3 Prompts"))
        heads = ["1 Introduction", "2 Method", "A.1 Data", "A.2 Models", "A.3 Prompts", "A.1 Data", "A.2 Models",
                 "A.3 Prompts"]
        doc, dropped = drop_toc_entries(text, heads)
        self.assertEqual(dropped, ["A.1 Data", "A.2 Models", "A.3 Prompts"])
        self.assertEqual(len(doc.sections), 5)
        for s in doc.sections:
            self.assertNotIn("#", s.id)
            self.assertTrue(text[s.char_start:].startswith(f"{s.title}\nReal text of {s.title}."), s.id)

    def test_a_truncated_contents_entry_does_not_take_the_id(self):
        # RFC 9110: the contents page was cut at "15.2.2." (a chunk boundary); the rest of the page is its body, and
        # the real "15.2.2. 101 Switching Protocols" became "15.2.2#2"
        from paper_structure.service import drop_toc_entries
        titles = ["15.1. Overview", "15.2. Informational", "15.2.1. Continue", "15.2.2. Switching Protocols",
                  "15.3. Successful", "15.3.1. OK", "15.3.2. Created"]
        body = "".join(f"{h}\nReal text of {h}. " + "More of it. " * 20 + "\n\n" for h in titles)
        text = "Contents\n" + "\n".join(titles[:3]) + "\n15.2.2.\n" + "\n".join(titles[3:]) + "\n\n" + body
        heads = titles[:3] + ["15.2.2."] + titles
        doc, dropped = drop_toc_entries(text, heads)
        self.assertEqual(dropped, titles[:3] + ["15.2.2."])
        self.assertEqual([s.title for s in doc.sections], titles)
        self.assertNotIn("#", "".join(s.id for s in doc.sections))

    def test_search_matches_across_line_breaks_and_hyphenation(self):
        # IEEE acceptance run 10-20: "KL divergence" missed "KL\ndivergence" in IV.B, and the agent answered from
        # the one other hit; PDF text also keeps hyphenation points inside words ("abbrevi￾ations")
        text = ("1 Intro\nWe align the scoring functions using KL\ndivergence here.\n\n"
                "2 Terms\nAmbiguous abbrevi￾ations and KL  divergence again.\n")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "doc.txt"
            f.write_text(text, encoding="utf-8")
            nav = Navigator(model="m", device="cpu", cache_dir=None, engine_factory=FakeEngine, config={})
            out = nav.search(f, "kl divergence")
            self.assertEqual((out["total_hits"], out["hits_per_section"]), (2, {"1": 1, "2": 1}))
            body = nav.outline(f) and f.read_text(encoding="utf-8")
            first = out["hits"][0]["char_start"]
            self.assertTrue(body[first:].startswith("KL\ndivergence"))
            self.assertEqual(nav.search(f, "abbreviations")["total_hits"], 1)
            self.assertEqual(nav.search(f, "  KL \n divergence ")["total_hits"], 2)   # the query's own spacing too
            self.assertEqual(nav.search(f, "divergence", section_id="2")["total_hits"], 1)
            self.assertIn("KL divergence here", out["hits"][0]["snippet"])

    def test_running_page_headers_are_dropped(self):
        # Large Multimodal Reasoning Models survey, 10-20: "3.2 Stage 2 ..." at the top of every page of 3.2 made
        # "3.2" 66 characters long (its text was in "3.2#2"), and 32 "REFERENCES" page headers became sections
        from paper_structure.service import tidy_headings
        para = lambda n: ("Real text. " * 30 + "\n") * n   # noqa: E731
        text = ("3.1 Setup\n" + para(2) + "3.2 Stage Two\n" + para(2) + "3.2 Stage Two\n" + para(1) + "3.2.1 Detail\n"
                + para(1) + "3.2 Stage Two\n" + para(1) + "4 Conclusion\n" + para(1)
                + "References\n" + "".join(f"[{n}] A cited work in a journal, with its authors and title.\n" * 20
                                           + "REFERENCES\n" for n in range(4)))   # one page of references each
        heads = ["3.1 Setup", "3.2 Stage Two", "3.2 Stage Two", "3.2.1 Detail", "3.2 Stage Two", "4 Conclusion",
                 "References", "REFERENCES", "REFERENCES", "REFERENCES", "REFERENCES"]
        doc, contents, repeats = tidy_headings(text, heads)
        self.assertEqual([s.id for s in doc.sections], ["3.1", "3.2", "3.2.1", "4", "references"])
        self.assertEqual((contents, len(repeats)), ([], 6))
        self.assertEqual(doc.sections[1].char_start, text.index("3.2 Stage Two"))
        self.assertTrue(doc.read_section("references").rstrip().endswith("REFERENCES"))   # the text stays

    def test_a_dropped_later_header_does_not_take_the_heading(self):
        # Grok review 10-20 (F1): with the later page header dropped but not blanked, the locator moved "1" onto it
        # (it prefers a match past the first 5% of the text over one inside it)
        from paper_structure.service import tidy_headings
        text = "1 Intro\n" + "intro " * 40 + "\n" + "1 Intro\n" + "page " * 40 + "\n" + "2 End\n" + "end " * 400
        doc, _, repeats = tidy_headings(text, ["1 Intro", "1 Intro", "2 End"])
        self.assertEqual(([s.char_start for s in doc.sections], repeats), ([0, text.index("2 End")], ["1 Intro"]))

    def test_a_page_header_set_into_a_sentence_gives_way(self):
        # Grok review 10-20 (F2): the header at the top of the page where 2 starts mid-page, the tail of 1 after it
        from paper_structure.service import tidy_headings
        text = ("1 Intro\n" + "This is the introduction. " * 60 + "And it goes on to a page break in the middle of\n"
                "2 Method\n" + "a sentence that ends on the next page. " * 6 + "\n2 Method\n" + "We describe the method. " * 40)
        doc, _, repeats = tidy_headings(text, ["1 Intro", "2 Method", "2 Method"])
        self.assertEqual([s.char_start for s in doc.sections], [0, text.index("\n2 Method\nWe") + 1])
        self.assertEqual(repeats, ["2 Method"])

    def test_three_proofs_in_a_row_are_not_page_headers(self):
        # Grok review 10-20 (F3): "Proof", "Proof", "Proof" a few lines apart were merged as a row of page headers
        from paper_structure.service import tidy_headings
        text = ("Intro\n" + "x " * 300 + "\nProof\n" + "p one " * 30 + "\nTheorem 2 holds.\nProof\n" + "p two " * 30
                + "\nProof\n" + "p three " * 30)
        doc, _, repeats = tidy_headings(text, ["Intro", "Proof", "Proof", "Proof"])
        self.assertEqual((len(doc.sections), repeats), (4, []))

    def test_search_finds_a_word_hyphenated_over_a_line_break(self):
        text = "1 Intro\nWe use abbrevi­\nations here and abbrevi￾ations there.\n"
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "doc.txt"
            f.write_text(text, encoding="utf-8")
            nav = Navigator(model="m", device="cpu", cache_dir=None, engine_factory=FakeEngine, config={})
            self.assertEqual(nav.search(f, "abbreviations")["total_hits"], 2)

    def test_structure_cache_follows_the_headings_cache(self):
        # Grok review 10-20 (F7): with the headings cache replaced, the old structure cache was still served
        import glob
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "doc.txt"
            f.write_text("1 Alpha\nbody one\n\n2 Beta\nbody two\n", encoding="utf-8")
            cache = Path(d) / "cache"
            nav = Navigator(model="m", device="cpu", cache_dir=cache, engine_factory=FakeEngine, config={})
            self.assertEqual([s["id"] for s in nav.outline(f)["sections"]], ["1", "2"])
            heads_file = next(p for p in cache.glob("*.json") if not p.name.endswith(".doc.json"))
            data = json.loads(heads_file.read_text(encoding="utf-8"))
            data["headings"] = ["1 Alpha"]   # as if predicted again with another result
            heads_file.write_text(json.dumps(data), encoding="utf-8")
            again = Navigator(model="m", device="cpu", cache_dir=cache, engine_factory=FakeEngine, config={})
            self.assertEqual([s["id"] for s in again.outline(f)["sections"]], ["1"])

    def test_a_single_contents_line_gives_way_to_the_heading(self):
        # DeepSeek-V3 report, 10-20: only "2.1.1 Multi-Head Latent Attention . . . 7" of its contents page was marked;
        # the first version of the repeat rule kept that line and dropped the real 2.1.1
        from paper_structure.service import tidy_headings
        para = "Real text. " * 30 + "\n"
        text = ("Contents\n2.1.1 Attention . . . . . . 7\n2.1.2 Experts . . . . . . 8\n\n" + para * 3 + "2.1.1 Attention\n"
                + para + "2.1.2 Experts\n" + para)
        doc, _, repeats = tidy_headings(text, ["2.1.1 Attention", "2.1.1 Attention", "2.1.2 Experts"])
        self.assertEqual([(s.id, s.char_start) for s in doc.sections],
                         [("2.1.1", text.index("2.1.1 Attention\n")), ("2.1.2", text.index("2.1.2 Experts\n"))])
        self.assertEqual(repeats, ["2.1.1 Attention"])

    def test_a_contents_line_without_page_numbers_gives_way_too(self):
        # contents lines in the first 5% of the text, without dot leaders: one followed by the next contents line,
        # one with nothing under it before the next marked heading
        from paper_structure.service import tidy_headings
        para = "Real text. " * 30 + "\n"
        text = ("Contents\n1 Introduction\n2 Method\n3 Results\n\n" + para * 5 + "1 Introduction\n" + para * 10
                + "2 Method\n" + para * 10 + "3 Results\n" + para * 10)
        heads = ["1 Introduction", "2 Method", "1 Introduction", "2 Method", "3 Results"]
        doc, _, repeats = tidy_headings(text, heads)
        self.assertEqual([(s.id, s.char_start) for s in doc.sections],
                         [("1", text.index("1 Introduction\nReal")), ("2", text.index("2 Method\nReal")),
                          ("3", text.index("3 Results\nReal"))])

    def test_an_early_real_heading_is_not_a_contents_line(self):
        # A3 (GPT-4 report), 10-20 scan: its own "1 Introduction" is at 0.3% of the text, and the system card
        # appended to it has another; the first version dropped the report's as a contents line
        from paper_structure.service import tidy_headings
        para = "Real text. " * 30 + "\n"
        text = ("1 Introduction\nThis report presents the model.\n" + para * 30 + "2 Scope\n" + para * 5 + "System Card\n"
                + para * 5 + "1 Introduction\nLarge language models are everywhere.\n" + para * 5)
        doc, _, repeats = tidy_headings(text, ["1 Introduction", "2 Scope", "System Card", "1 Introduction"])
        self.assertEqual(repeats, [])
        self.assertEqual(doc.sections[0].char_start, 0)

    def test_a_dropped_earlier_line_is_not_used_again(self):
        # the locator prefers the earliest of equal matches: an identical contents line past the first 5% would get
        # the kept heading back unless it is blanked
        from paper_structure.service import tidy_headings
        para = "Real text. " * 30 + "\n"
        text = ("1 Intro\n" + para * 30 + "Appendix contents\nA.1 Data\nA.2 Models\n\n" + "A.1 Data\n" + para
                + "A.2 Models\n" + para)
        doc, _, repeats = tidy_headings(text, ["1 Intro", "A.1 Data", "A.1 Data", "A.2 Models"])
        self.assertEqual([(s.id, s.char_start) for s in doc.sections],
                         [("1", 0), ("A.1", text.index("A.1 Data\nReal")), ("A.2", text.index("A.2 Models\nReal"))])
        self.assertEqual(repeats, ["A.1 Data"])

    def test_an_empty_parent_heading_is_not_a_page_header(self):
        # "2 Method" directly followed by "2.1 Data" has no body of its own; its running header later does
        from paper_structure.service import tidy_headings
        para = "Real text. " * 30 + "\n"
        text = "1 Intro\n" + para * 30 + "2 Method\n2.1 Data\n" + para * 3 + "2 Method\n" + para * 3
        doc, _, repeats = tidy_headings(text, ["1 Intro", "2 Method", "2.1 Data", "2 Method"])
        self.assertEqual([(s.id, s.char_start) for s in doc.sections],
                         [("1", 0), ("2", text.index("2 Method")), ("2.1", text.index("2.1 Data"))])

    def test_template_headings_repeated_per_example_are_kept(self):
        # D2 (LLM research ideas), 10-20 scan: every example idea in the appendix has "4. Step-by-Step Experiment
        # Plan"; the first version of the repeat rule took the later ones for page headers and merged them away
        from paper_structure.service import tidy_headings
        para = "Real text. " * 30 + "\n"
        text = ("P Example Idea One\n" + para + "1. Problem Statement\n" + para + "4. Step-by-Step Experiment Plan\n" + para
                + "Q Example Idea Two\n" + para + "1. Problem Statement\n" + para + "4. Step-by-Step Experiment Plan\n" + para)
        heads = ["P Example Idea One", "1. Problem Statement", "4. Step-by-Step Experiment Plan", "Q Example Idea Two",
                 "1. Problem Statement", "4. Step-by-Step Experiment Plan"]
        doc, _, repeats = tidy_headings(text, heads)
        self.assertEqual(repeats, [])
        self.assertEqual([s.id for s in doc.sections], ["P", "1", "4", "Q", "1#2", "4#2"])

    def test_a_marked_caption_does_not_separate_page_headers(self):
        # Large Multimodal Reasoning survey: a figure caption the model marked sat between "3.1 ..." and its page
        # header; an unnumbered heading is not a sibling that starts a new instance
        from paper_structure.service import tidy_headings
        para = "Real text. " * 30 + "\n"
        text = ("Front\n" + para * 3 + "3.1 Stage One\n" + para + "Stage 1 figure caption\n" + para + "3.1 Stage One\n" + para
                + "3.2 Stage Two\n" + para)
        doc, _, repeats = tidy_headings(text, ["3.1 Stage One", "Stage 1 figure caption", "3.1 Stage One", "3.2 Stage Two"])
        self.assertEqual(repeats, ["3.1 Stage One"])

    def test_a_second_document_in_the_file_keeps_its_sections(self):
        from paper_structure.service import tidy_headings
        para = "Real text. " * 30 + "\n"
        text = ("A Report\nAuthors\n" + para + "1 Introduction\n" + para + "2 Results\n" + para + "System Card\n" + para
                + "1 Introduction\n" + para + "2 Observed Challenges\n" + para)   # (front matter: past the first 5%)
        doc, _, repeats = tidy_headings(text, ["1 Introduction", "2 Results", "System Card", "1 Introduction",
                                               "2 Observed Challenges"])
        self.assertEqual((repeats, len(doc.sections)), ([], 5))

    def test_same_title_under_different_ieee_sections_is_kept(self):
        from paper_structure.service import tidy_headings
        para = "Real text. " * 30 + "\n"
        text = "I. INTRODUCTION\n" + para + "II. METHOD\n" + para + "A. Formulation\n" + para + "III. THEORY\n" + para \
            + "A. Formulation\n" + para
        doc, _, repeats = tidy_headings(text, ["I. INTRODUCTION", "II. METHOD", "A. Formulation", "III. THEORY",
                                               "A. Formulation"])
        self.assertEqual([s.id for s in doc.sections], ["I", "II", "II.A", "III", "III.A"])
        self.assertEqual(repeats, [])

    def test_outline_counts_repeated_headings(self):
        text = "1 Intro\n" + "Real text. " * 30 + "\n1 Intro\n" + "Real text. " * 30 + "\n2 End\n" + "Real. " * 30
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "doc.txt"
            f.write_text(text, encoding="utf-8")
            out = Navigator(model="m", device="cpu", cache_dir=None, engine_factory=FakeEngine, config={}).outline(f)
        self.assertEqual([s["id"] for s in out["sections"]], ["1", "2"])
        self.assertEqual(out["repeated_headings_skipped"], 1)

    def test_repeated_real_headings_are_kept(self):
        text = ("Part A\n\nResults\n" + "First results text. " * 10 + "\n\nPart B\n\nResults\n"
                + "Second results text. " * 10 + "\n")
        from paper_structure.service import drop_toc_entries
        doc, dropped = drop_toc_entries(text, ["Results", "Results"])
        self.assertEqual(dropped, [])
        self.assertEqual([s.id for s in doc.sections], ["results", "results#2"])

    def test_one_short_real_section_with_a_later_namesake_is_kept(self):
        # Codex review 10-20: "Results\nOK." was dropped as a contents entry
        from paper_structure.service import drop_toc_entries
        text = "Results\nOK.\n\nResults\n" + "Substantial second section body. " * 5
        doc, dropped = drop_toc_entries(text, ["Results", "Results"])
        self.assertEqual(dropped, [])
        self.assertEqual([s.char_start for s in doc.sections], [0, 13])

    def test_structure_built_by_older_code_is_removed(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "doc.txt"
            f.write_text("1 Alpha\nbody one\n\n2 Beta\nbody two\n", encoding="utf-8")
            cache = Path(d) / "cache"
            nav = Navigator(model="m", device="cpu", cache_dir=cache, engine_factory=FakeEngine, config={})
            sha = hashlib.sha256(f.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            cache.mkdir()
            stale = cache / f"{sha[:32]}_{nav._model_key()}_0000000000.doc.json"
            stale.write_text("{}", encoding="utf-8")
            other = cache / f"{'0' * 32}_{nav._model_key()}_0000000000.doc.json"   # another file's: kept
            other.write_text("{}", encoding="utf-8")
            nav.outline(f)
            self.assertFalse(stale.exists())
            self.assertTrue(other.exists())
            self.assertEqual(len(list(cache.glob(f"{sha[:32]}_*.doc.json"))), 1)

    def test_content_change_with_same_size_and_mtime_is_seen(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "doc.txt"
            f.write_text("1 Alpha\nbody one\n\n2 Beta\nbody two\n", encoding="utf-8")
            st = f.stat()
            cache = Path(d) / "cache"
            Navigator(model="m", device="cpu", cache_dir=cache, engine_factory=FakeEngine, config={}).outline(f)
            f.write_text("1 Gamma\nbody one\n\n2 Zeta\nbody two\n", encoding="utf-8")   # same length
            os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
            self.assertEqual(f.stat().st_size, st.st_size)
            out = Navigator(model="m", device="cpu", cache_dir=cache, engine_factory=FakeEngine, config={}).outline(f)
            self.assertEqual([s["title"] for s in out["sections"]], ["1 Gamma", "2 Zeta"])

    def test_processes_sharing_a_cache_keep_each_others_entries(self):
        with tempfile.TemporaryDirectory() as d:
            cache = Path(d) / "cache"
            files = []
            for i in range(3):
                f = Path(d) / f"doc{i}.txt"
                f.write_text(f"1 Part{i}\nbody\n\n2 More{i}\nbody\n", encoding="utf-8")
                files.append(f)
                Navigator(model="m", device="cpu", cache_dir=cache, engine_factory=FakeEngine, config={}).outline(f)
            later = Navigator(model="m", device="cpu", cache_dir=cache, engine_factory=explode, config={})
            self.assertTrue(all(later.outline(f)["from_cache"] for f in files))
            self.assertFalse((cache / "files.json").exists())   # no shared read-modify-write index

    def test_hits_per_section_counts_every_hit(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "doc.txt"
            f.write_text("1 Alpha\n" + "x " * 6 + "\n\n2 Beta\nx\n", encoding="utf-8")
            nav = Navigator(model="m", device="cpu", cache_dir=None, engine_factory=FakeEngine, config={})
            r = nav.search(f, "x", max_results=2)
            self.assertEqual((r["total_hits"], r["returned"]), (7, 2))
            self.assertEqual(r["hits_per_section"], {"1": 6, "2": 1})

    def test_mcp_rejects_bad_messages_and_keeps_running(self):
        import io
        from paper_structure.mcp_stdio import Server
        out = io.BytesIO()
        server = Server(Navigator(model="m", device="cpu", cache_dir=None, engine_factory=FakeEngine, config={}), out)
        lines = [b"[]", b"42", b'{"jsonrpc":"2.0","id":null,"method":"ping"}',
                 b'{"jsonrpc":"2.0","id":2,"method":"initialize","params":["x"]}',
                 b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":["x"]}}',
                 b'{"jsonrpc":"2.0","id":4,"result":{}}', b'{"jsonrpc":"2.0","id":5,"method":"ping"}']
        server.serve(io.BytesIO(b"\n".join(lines) + b"\n"))
        replies = [json.loads(ln) for ln in out.getvalue().decode("utf-8").splitlines()]
        self.assertEqual([r.get("error", {}).get("code") for r in replies], [-32600, -32600, None, -32602, -32602, None])
        self.assertEqual([r["id"] for r in replies], [None, None, None, 2, 3, 5])

    def test_locator_diagnostics_stay_off_stdout(self):
        import io
        from contextlib import redirect_stdout
        from paper_structure.locate import locate_headings
        text = "\n".join(["Results"] * 80 + ["body"])
        out = io.StringIO()
        with redirect_stdout(out):
            locate_headings(text, ["Results"])
        self.assertEqual(out.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
