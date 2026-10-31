"""Page numbers (pdf.join_pages / extract_pdf_pages, Navigator.page_starts) and opening a local file in place
(App.submit_path, POST /api/open_path), with a fake model and a small hand-written PDF."""

from __future__ import annotations

import json
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from paper_structure.pdf import extract_pdf_pages, extract_pdf_text, join_pages, normalise, page_of
from paper_structure.service import Navigator

from test_app import FakeEngine, TestApp


def make_pdf(pages: list[list[str]]) -> bytes:
    """A minimal valid PDF: one Helvetica text line per string, one page per list."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for lines in pages:
        body = "BT /F1 12 Tf 14 TL 72 720 Td " + " ".join(
            "(" + ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ") Tj T*" for ln in lines) + " ET"
        objs.append(f"<< /Length {len(body)} >>\nstream\n{body}\nendstream")
        content = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
                    f"/Contents {content} 0 R >>")
        kids.append(len(objs))
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


PAGES = [["Paper title", "Abstract", "We study things.", "1 Introduction", "Things matter."],
         ["2 Method", "We do things.", "2.1 Data", "We collect data."],
         ["3 Results", "It works better than the baseline."]]


class TestJoinPages(unittest.TestCase):
    def test_same_text_as_before_and_page_starts(self):
        for pages in (["a\r\nb", "c\r\nd\r\n", "e"], ["x\r", "\ny\r\nz", "\r\nw"], ["", "q", ""], ["only"]):
            text, starts = join_pages(pages)
            self.assertEqual(text, normalise("\n".join(pages)), pages)
            for i, p in enumerate(pages):
                # the page's own text (normalised alone) starts at its position; a trailing "\r" of the page before
                # pairs with the joining newline, not with this page's first character
                self.assertTrue(text[starts[i]:].startswith(normalise(p)), (pages, i, starts, text))
            self.assertEqual(page_of(starts, 0), 1)
            self.assertEqual(page_of(starts, len(text) - 1), max(i + 1 for i, s in enumerate(starts) if s <= len(text) - 1))


@unittest.skipUnless(find_spec("pypdfium2"), "pypdfium2 not installed")
class TestPdfPages(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.pdf = self.dir / "paper.pdf"
        self.pdf.write_bytes(make_pdf(PAGES))

    def tearDown(self):
        self._tmp.cleanup()

    def test_pages_text_is_the_benchmark_text(self):
        text, starts = extract_pdf_pages(self.pdf)
        self.assertEqual(text, extract_pdf_text(self.pdf))
        self.assertEqual(len(starts), 3)
        self.assertIn("2 Method", text[starts[1]:starts[2]])

    def test_outline_read_search_carry_pages(self):
        nav = Navigator(model="fake", device="cpu", cache_dir=self.dir / "cache", engine_factory=FakeEngine, config={})
        out = nav.outline(self.pdf)
        self.assertEqual(out["pages"], 3)
        pages = {s["id"]: s["page"] for s in out["sections"]}
        self.assertEqual(pages, {"abstract": 1, "1": 1, "2": 2, "2.1": 2, "3": 3})
        self.assertEqual(nav.read(self.pdf, "2.1")["page"], 2)
        self.assertEqual(nav.search(self.pdf, "baseline")["hits"][0]["page"], 3)
        cached = list((self.dir / "cache").glob("*.pages.json"))
        self.assertEqual(len(cached), 1)
        # a new navigator reads the cached page starts (no PDF parsing needed) and gives the same pages
        nav2 = Navigator(model="fake", device="cpu", cache_dir=self.dir / "cache", engine_factory=FakeEngine, config={})
        self.assertEqual({s["id"]: s["page"] for s in nav2.outline(self.pdf)["sections"]}, pages)

    def test_text_files_have_no_pages(self):
        txt = self.dir / "paper.txt"
        txt.write_text("Abstract\nx\n1 Introduction\ny\n", encoding="utf-8")
        nav = Navigator(model="fake", device="cpu", cache_dir=self.dir / "cache", engine_factory=FakeEngine, config={})
        out = nav.outline(txt)
        self.assertNotIn("pages", out)
        self.assertTrue(all("page" not in s for s in out["sections"]))


class TestOpenPath(TestApp):
    @unittest.skipUnless(find_spec("pypdfium2"), "pypdfium2 not installed")
    def test_open_path_reads_in_place(self):
        pdf = Path(self._tmp.name) / "zotero storage" / "论文.pdf"
        pdf.parent.mkdir()
        pdf.write_bytes(make_pdf(PAGES))
        code, job = self.post_json("/api/open_path", {"path": str(pdf)})
        self.assertEqual(code, 200, job)
        view = self.wait(job["job_id"])
        self.assertEqual(view["status"], "done", view.get("error"))
        self.assertEqual(view["name"], "论文.pdf")
        self.assertEqual(Path(view["path"]), pdf.resolve())
        self.assertEqual(view["pages"], 3)
        self.assertEqual([(s["id"], s["page"]) for s in view["sections"]],
                         [("abstract", 1), ("1", 1), ("2", 2), ("2.1", 2), ("3", 3)])
        self.assertFalse((self.app.uploads).exists() and any(self.app.uploads.rglob("论文.pdf")))   # not copied

    def test_open_path_errors(self):
        self.assertEqual(self.post_json("/api/open_path", {"path": str(Path(self._tmp.name) / "missing.pdf")})[0], 400)
        self.assertEqual(self.post_json("/api/open_path", {"file": "x"})[0], 400)
        self.assertEqual(self.post_json("/api/open_path", {"path": self._tmp.name})[0], 400)   # a folder


if __name__ == "__main__":
    unittest.main()
