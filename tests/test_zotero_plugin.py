"""The Zotero plugin's extraction end to end, without Zotero: a real local service (fake model) and the plugin's own
JavaScript run by Node against a fake Zotero object (tests/zotero/run_plugin_flow.mjs). Skipped without Node."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from importlib.util import find_spec
from pathlib import Path

from test_app import TestApp
from test_pages import PAGES, make_pdf

HERE = Path(__file__).resolve().parent
NODE = shutil.which("node")


@unittest.skipIf(NODE is None, "node not installed")
class TestZoteroPluginFlow(TestApp):
    def run_flow(self, pdf: Path) -> dict:
        out = subprocess.run([NODE, str(HERE / "zotero" / "run_plugin_flow.mjs"), self.base, str(pdf)],
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout.strip().splitlines()[-1])

    @unittest.skipUnless(find_spec("pypdfium2"), "pypdfium2 not installed")
    def test_menu_flow_writes_one_outline_note(self):
        pdf = Path(self._tmp.name) / "storage" / "ABCD" / "paper.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(make_pdf(PAGES))
        res = self.run_flow(pdf)
        self.assertEqual(res["notes_after_first"], 1)
        self.assertEqual(res["notes_after_second"], 1, "the second run replaces the note")
        html = res["html"]
        self.assertIn("章节结构（paper-structure）", html)
        self.assertIn(f'href="zotero://open-pdf/library/items/{res["pdfKey"]}?page=2">2 Method</a>', html)
        self.assertIn("3 页", html)
        texts = [ln.get("text", "") for ln in res["lines"]]
        self.assertTrue(any(t.startswith("A paper：5 节") for t in texts), texts)
        self.assertTrue(any("No PDF：没有 PDF 附件" in t for t in texts), texts)
        self.assertTrue(any("（缓存）" in t for t in texts), "the second run is served from the cache")

    def test_service_down(self):
        self.server.shutdown()
        self.server.server_close()
        pdf = Path(self._tmp.name) / "p.pdf"
        pdf.write_bytes(make_pdf(PAGES))
        res = self.run_flow(pdf)
        self.assertTrue(any("没连上本地服务" in ln.get("description", "") for ln in res["lines"]), res["lines"])
        self.assertEqual(res["notes_after_first"], 0)
        self.server = type("S", (), {"shutdown": lambda s: None, "server_close": lambda s: None})()   # for tearDown


if __name__ == "__main__":
    unittest.main()
