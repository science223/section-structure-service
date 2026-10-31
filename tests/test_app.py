"""The local web app end to end over real HTTP (127.0.0.1, ephemeral port) with a fake model."""

from __future__ import annotations

import json
import re
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

from paper_structure.app import App, make_server, read_input
from paper_structure.infer import ChunkPrediction

TEXT = ("Paper title\n\nAbstract\nWe study things.\n\n1 Introduction\nThings matter. Baselines are weak.\n\n"
        "2 Method\nWe do things.\n\n2.1 Data\nWe collect data.\n\n3 Results\nIt works better than the baseline.\n")


class FakeEngine:
    """Marks every line that looks like a numbered heading or 'Abstract'; counts calls."""
    tokenizer = None

    def __init__(self):
        self.calls = 0

    def predict_chunk(self, chunk):
        self.calls += 1
        heads = [ln for ln in chunk["text"].split("\n") if re.fullmatch(r"(\d+(\.\d+)* [A-Z]\w+|Abstract)", ln)]
        return ChunkPrediction(chunk["chunk_id"], "\n".join(f"###H### {h}" for h in heads), heads)


class TestApp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.engine = FakeEngine()
        self.app = App(model="fake-model", device="cpu", cache_dir=Path(self._tmp.name),
                       engine_factory=lambda: self.engine, config={})
        self.server = make_server(self.app, port=0)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self._tmp.cleanup()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers, e.read()

    def get_json(self, path):
        code, _, body = self.get(path)
        return code, json.loads(body)

    def open_file(self, name, data):
        req = urllib.request.Request(self.base + "/api/open", data=data, method="POST",
                                     headers={"X-Filename": quote(name)})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def wait(self, job_id):
        for _ in range(200):
            code, view = self.get_json(f"/api/jobs/{job_id}")
            if view["status"] in ("done", "error"):
                return view
            time.sleep(0.05)
        self.fail("job did not finish")

    def test_page_and_info(self):
        code, headers, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertEqual(self.get_json("/api/info")[1]["device"], "cpu")

    def test_open_read_search_export(self):
        job = self.open_file("paper.txt", TEXT.encode("utf-8"))
        view = self.wait(job["job_id"])
        self.assertEqual(view["status"], "done", view.get("error"))
        self.assertEqual([s["id"] for s in view["sections"]], ["abstract", "1", "2", "2.1", "3"])
        self.assertEqual(view["tool_call"]["name"], "get_outline")
        self.assertGreater(view["stats"]["tokens_full"], view["sections"][1]["tokens"])
        self.assertEqual(self.get_json(f"/api/jobs/{job['job_id']}/text")[1]["text"], TEXT)
        code, read = self.get_json(f"/api/jobs/{job['job_id']}/read?section=2&descendants=1")
        self.assertEqual(code, 200)
        self.assertTrue(read["text"].startswith("2 Method") and "2.1 Data" in read["text"])
        self.assertTrue(read["tool_call"]["arguments"]["include_subsections"])
        code, direct = self.get_json(f"/api/jobs/{job['job_id']}/read?section=2")
        self.assertNotIn("2.1 Data", direct["text"])
        code, hits = self.get_json(f"/api/jobs/{job['job_id']}/search?section=3&q=BASELINE")
        self.assertEqual(len(hits["hits"]), 1)
        self.assertEqual(TEXT[hits["hits"][0]["char_start"]:hits["hits"][0]["char_end"]].lower(), "baseline")
        code, headers, body = self.get(f"/api/jobs/{job['job_id']}/export?format=md")
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertIn("  - 2.1 Data", body.decode("utf-8"))
        code, headers, body = self.get(f"/api/jobs/{job['job_id']}/export?format=json")
        self.assertEqual(json.loads(body)["sections"][0]["id"], "abstract")

    def test_cache_skips_the_model(self):
        first = self.wait(self.open_file("a.txt", TEXT.encode("utf-8"))["job_id"])
        calls = self.engine.calls
        second = self.wait(self.open_file("copy of a.txt", TEXT.encode("utf-8"))["job_id"])
        self.assertEqual(self.engine.calls, calls)
        self.assertTrue(second["cached"] and not first["cached"])
        self.assertEqual(second["sections"], first["sections"])
        self.assertEqual(self.get_json("/api/jobs")[1]["jobs"][0]["name"], "copy of a.txt")   # newest first

    def test_errors(self):
        job = self.wait(self.open_file("latin1.txt", "caf\xe9".encode("latin-1"))["job_id"])
        self.assertEqual(job["status"], "error")
        self.assertIn("neither a PDF nor UTF-8", job["error"])
        ok = self.wait(self.open_file("p.txt", TEXT.encode("utf-8"))["job_id"])
        code, body = self.get_json(f"/api/jobs/{ok['job_id']}/read?section=9.9")
        self.assertEqual(code, 404)
        self.assertIn("Unknown section id '9.9'", body["error"])
        self.assertEqual(self.get_json("/api/jobs/j999")[0], 404)
        self.assertEqual(self.get_json(f"/api/jobs/{ok['job_id']}/export?format=docx")[0], 400)
        self.assertEqual(self.get_json(f"/api/jobs/{ok['job_id']}/search?section=1&q=")[0], 400)

    def post_json(self, path, obj):
        req = urllib.request.Request(self.base + path, data=json.dumps(obj).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_tools_over_http_and_openapi(self):
        f = Path(self._tmp.name) / "doc.txt"
        f.write_text(TEXT, encoding="utf-8")
        code, out = self.post_json("/v1/tools/get_outline", {"file_path": str(f)})
        self.assertEqual(code, 200)
        self.assertEqual([s["id"] for s in out["sections"]], ["abstract", "1", "2", "2.1", "3"])
        code, out = self.post_json("/v1/tools/search_document", {"file_path": str(f), "query": "baseline"})
        self.assertEqual((code, out["hits_per_section"]), (200, {"1": 1, "3": 1}))
        code, out = self.post_json("/v1/tools/read_section", {"file_path": str(f), "section_id": "7"})
        self.assertEqual((code, out["error"]["code"]), (400, "unknown_section"))
        code, out = self.post_json("/v1/tools/nope", {})
        self.assertEqual((code, out["error"]["code"]), (400, "bad_argument"))
        code, spec = self.get_json("/openapi.json")
        self.assertEqual(spec["openapi"], "3.1.0")
        self.assertEqual(sorted(spec["paths"]), ["/v1/tools/get_outline", "/v1/tools/read_section",
                                                 "/v1/tools/search_document"])
        self.assertTrue(spec["servers"][0]["url"].startswith("http://127.0.0.1:"))
        op = spec["paths"]["/v1/tools/read_section"]["post"]
        self.assertEqual(op["requestBody"]["content"]["application/json"]["schema"]["required"], ["file_path", "section_id"])

    def test_tool_body_size_is_limited(self):
        from paper_structure.app import MAX_TOOL_BODY
        code, out = self.post_json("/v1/tools/get_outline", {"file_path": "x" * MAX_TOOL_BODY})
        self.assertEqual(code, 413)
        self.assertEqual(out["error"]["code"], "bad_argument")

    def test_too_large_upload_gets_an_answer(self):
        from unittest import mock
        with mock.patch("paper_structure.app.MAX_UPLOAD_BYTES", 1000):
            req = urllib.request.Request(self.base + "/api/open", data=b"x" * 200_000, method="POST",
                                         headers={"X-Filename": "big.txt"})
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=10)
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("file larger than", json.loads(cm.exception.read())["error"])

    def test_upload_name_is_reduced_to_a_file_name(self):
        job = self.open_file("..\\..\\secret/dir/paper.txt", TEXT.encode("utf-8"))
        self.assertEqual(job["name"], "paper.txt")

    def test_server_binds_localhost_only(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")


class TestConfig(unittest.TestCase):
    def test_model_from_config_after_flag_and_env(self):
        import os
        from unittest import mock
        from paper_structure.app import load_config
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "config.json"
            cfg.write_text(json.dumps({"model": "D:/models/学术/m", "device": "cpu"}), encoding="utf-8")
            conf = load_config(cfg)
            self.assertEqual(conf["model"], "D:/models/学术/m")
            self.assertEqual(load_config(Path(d) / "missing.json"), {})
            (Path(d) / "bad.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(load_config(Path(d) / "bad.json"), {})
        with mock.patch.dict(os.environ, {"PAPER_STRUCTURE_MODEL": "", "PAPER_STRUCTURE_DEVICE": ""}):
            app = App(cache_dir=None, engine_factory=lambda: None, config=conf)
            self.assertEqual((app.model, app.device), ("D:/models/学术/m", "cpu"))
            app = App(model="flag", device="cuda", cache_dir=None, engine_factory=lambda: None, config=conf)
            self.assertEqual((app.model, app.device), ("flag", "cuda"))
        with mock.patch.dict(os.environ, {"PAPER_STRUCTURE_MODEL": "env", "PAPER_STRUCTURE_DEVICE": ""}):
            self.assertEqual(App(cache_dir=None, engine_factory=lambda: None, config=conf).model, "env")


class TestReadInput(unittest.TestCase):
    def test_text_and_bom_and_crlf(self):
        self.assertEqual(read_input("x.txt", "﻿a\r\nb\rc".encode("utf-8")), "a\nb\nc")

    def test_pdf_header_goes_to_the_pdf_reader(self):
        try:
            import pypdfium2  # noqa: F401
        except ImportError:
            self.skipTest("pypdfium2 not installed")
        with self.assertRaises(Exception):
            read_input("broken.pdf", b"%PDF-1.7 not really a pdf")


if __name__ == "__main__":
    unittest.main()
